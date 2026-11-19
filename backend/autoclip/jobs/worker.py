"""Subprocess entry point for one pipeline job.

Whisper/CTranslate2 (and occasionally MediaPipe) can ACCESS_VIOLATION on
Windows. Running the pipeline out-of-process keeps the web server alive when
that happens, and surfaces a normal failed job instead of a dead UI.
"""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

log = logging.getLogger(__name__)


def _cancel_flag(job_id: str) -> Path:
    from .. import paths

    return paths.job_work_dir(job_id) / "cancel.flag"


def run_job(job_id: str) -> int:
    """Execute ``job_id`` and return a process exit code."""
    from .. import db
    from ..db import store
    from ..pipeline.runner import JobCancelled, PipelineRunner

    db.init()
    job = store.get_job(job_id)
    if job is None:
        log.error("Job %s not found.", job_id)
        return 2

    source = store.get_source(job.source_id)
    if source is None:
        store.update_job(job_id, status="failed", error="Source not found.")
        return 2

    flag = _cancel_flag(job_id)
    flag.unlink(missing_ok=True)

    def is_cancelled() -> bool:
        return flag.exists()

    runner = PipelineRunner(job, source, is_cancelled=is_cancelled)
    try:
        asyncio.run(runner.run())
    except JobCancelled:
        log.info("Job %s cancelled.", job_id)
        return 0
    except Exception as exc:
        log.exception("Job %s failed.", job_id)
        # PipelineRunner already persisted failed status for most errors.
        current = store.get_job(job_id)
        if current is not None and current.status == "running":
            store.update_job(job_id, status="failed", error=str(exc))
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = argv if argv is not None else sys.argv[1:]
    if not args:
        print("usage: python -m autoclip.jobs.worker <job_id>", file=sys.stderr)
        return 2
    return run_job(args[0])


if __name__ == "__main__":
    raise SystemExit(main())
