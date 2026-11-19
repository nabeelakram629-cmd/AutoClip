"""In-process job queue that runs each pipeline job in a subprocess.

One video is processed at a time, FIFO. That's a deliberate constraint rather
than a limitation: transcription, face detection, and encoding each want the
whole GPU, and running two jobs concurrently on a 6 GB card makes both slower
than running them in sequence.

The queue is durable because it lives in SQLite — a crash mid-job leaves the row
as ``running``, and startup requeues it so the resumable stages pick up where
they left off.

Jobs run in a child process so a native crash in Whisper/CTranslate2 (common on
some Windows setups as ACCESS_VIOLATION) fails the job instead of killing the
web server.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

from .. import paths
from ..db import store
from ..db.models import Job, utcnow
from .events import Event, broker

log = logging.getLogger(__name__)

#: Windows STATUS_ACCESS_VIOLATION — what CTranslate2 often dies with here.
_ACCESS_VIOLATION = 0xC0000005


@dataclass
class QueueStatus:
    running_job_id: str | None
    queued: int


class JobQueue:
    """Runs queued jobs one at a time."""

    def __init__(self) -> None:
        self._task: asyncio.Task | None = None
        # Created in start(), not here: an asyncio.Event binds to the loop that
        # is running when it is constructed, and this object is a module-level
        # singleton built at import time. Constructing it eagerly makes every
        # subsequent event loop in the process fail with "bound to a different
        # event loop".
        self._wakeup: asyncio.Event | None = None
        self._running_job_id: str | None = None
        self._cancel_requested: set[str] = set()
        self._stopping = False
        self._worker_proc: asyncio.subprocess.Process | None = None

    # -- lifecycle ---------------------------------------------------------

    async def start(self) -> None:
        """Start the worker and requeue anything left running by a crash."""
        self._requeue_interrupted()
        self._stopping = False
        self._wakeup = asyncio.Event()
        self._task = asyncio.create_task(self._worker(), name="autoclip-job-queue")
        self.notify()

    async def stop(self) -> None:
        self._stopping = True
        self.notify()
        if self._worker_proc is not None and self._worker_proc.returncode is None:
            with contextlib.suppress(ProcessLookupError):
                self._worker_proc.kill()
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None
        self._wakeup = None

    def notify(self) -> None:
        """Wake the worker — call after enqueuing a job.

        A no-op when the worker isn't running, so enqueuing still works in a
        process started without one.
        """
        if self._wakeup is not None:
            self._wakeup.set()

    def _requeue_interrupted(self) -> None:
        interrupted = store.list_jobs(limit=100, status="running")
        for job in interrupted:
            log.info("Requeuing job %s, interrupted by a restart.", job.id)
            store.update_job(job.id, status="queued", error=None)

    # -- control -----------------------------------------------------------

    def cancel(self, job_id: str) -> bool:
        """Request cancellation. Returns True if the job was running or queued."""
        job = store.get_job(job_id)
        if job is None or job.status in ("done", "failed", "cancelled"):
            return False

        self._cancel_requested.add(job_id)
        _cancel_flag(job_id).write_text("1", encoding="utf-8")
        if job.status == "queued":
            store.update_job(job_id, status="cancelled")
            broker.publish(Event(type="cancelled", job_id=job_id))
        elif self._running_job_id == job_id and self._worker_proc is not None:
            with contextlib.suppress(ProcessLookupError):
                self._worker_proc.kill()
        return True

    def is_cancelled(self, job_id: str) -> bool:
        return job_id in self._cancel_requested

    def status(self) -> QueueStatus:
        return QueueStatus(
            running_job_id=self._running_job_id,
            queued=len(store.list_jobs(limit=100, status="queued")),
        )

    # -- worker ------------------------------------------------------------

    async def _worker(self) -> None:
        wakeup = self._wakeup
        assert wakeup is not None

        while not self._stopping:
            job = await asyncio.to_thread(store.next_queued_job)
            if job is None:
                wakeup.clear()
                await wakeup.wait()
                continue

            await self._run_one(job)

    async def _run_one(self, job: Job) -> None:
        source = await asyncio.to_thread(store.get_source, job.source_id)
        if source is None:
            store.update_job(job.id, status="failed", error="Source not found.")
            broker.publish(Event(type="failed", job_id=job.id, data={"error": "Source not found."}))
            return

        self._running_job_id = job.id
        broker.publish(Event(type="started", job_id=job.id))
        log.info("Starting job %s (%s) in a subprocess.", job.id, source.title)

        paths.job_work_dir(job.id).mkdir(parents=True, exist_ok=True)
        _cancel_flag(job.id).unlink(missing_ok=True)

        last_stage = job.current_stage
        last_progress = job.progress

        try:
            proc = await asyncio.create_subprocess_exec(
                sys.executable,
                "-m",
                "autoclip.jobs.worker",
                job.id,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
            self._worker_proc = proc

            async def pump_output() -> None:
                assert proc.stdout is not None
                while line := await proc.stdout.readline():
                    log.info("[job %s] %s", job.id, line.decode("utf-8", "replace").rstrip())

            output_task = asyncio.create_task(pump_output())

            while proc.returncode is None:
                current = await asyncio.to_thread(store.get_job, job.id)
                if current is not None and (
                    current.current_stage != last_stage or current.progress != last_progress
                ):
                    last_stage = current.current_stage
                    last_progress = current.progress
                    broker.publish(
                        Event(
                            type="progress",
                            job_id=job.id,
                            data={
                                "stage": current.current_stage or "",
                                "stage_progress": 0.0,
                                "overall": current.progress,
                                "message": current.current_stage or "",
                            },
                        )
                    )
                try:
                    await asyncio.wait_for(proc.wait(), timeout=0.5)
                except TimeoutError:
                    continue

            await output_task
            code = proc.returncode or 0
            current = await asyncio.to_thread(store.get_job, job.id)

            if job.id in self._cancel_requested or (current and current.status == "cancelled"):
                if current is None or current.status != "cancelled":
                    await asyncio.to_thread(
                        store.update_job,
                        job.id,
                        status="cancelled",
                        finished_at=utcnow(),
                    )
                broker.publish(Event(type="cancelled", job_id=job.id))
                log.info("Job %s cancelled.", job.id)
            elif current is not None and current.status == "done":
                broker.publish(Event(type="done", job_id=job.id))
                log.info("Job %s finished.", job.id)
            elif current is not None and current.status == "failed":
                broker.publish(
                    Event(type="failed", job_id=job.id, data={"error": current.error or ""})
                )
                log.error("Job %s failed: %s", job.id, current.error)
            else:
                error = _explain_worker_exit(code)
                await asyncio.to_thread(
                    store.update_job,
                    job.id,
                    status="failed",
                    error=error,
                    finished_at=utcnow(),
                )
                broker.publish(Event(type="failed", job_id=job.id, data={"error": error}))
                log.error("Job %s aborted (exit %s): %s", job.id, code, error)
        finally:
            self._worker_proc = None
            self._running_job_id = None
            self._cancel_requested.discard(job.id)
            _cancel_flag(job.id).unlink(missing_ok=True)


def _cancel_flag(job_id: str) -> Path:
    return paths.job_work_dir(job_id) / "cancel.flag"


def _explain_worker_exit(code: int) -> str:
    unsigned = code & 0xFFFFFFFF
    if unsigned == _ACCESS_VIOLATION or code == -1073741819:
        return (
            "Transcription crashed inside Whisper/CTranslate2 (Windows access violation). "
            "Try Settings → Whisper model `tiny` or `base`, set compute type to `float32`, "
            "or reinstall with `uv pip install --reinstall ctranslate2 faster-whisper`."
        )
    return f"Pipeline worker exited unexpectedly (code {code})."


#: Process-wide queue.
queue = JobQueue()
