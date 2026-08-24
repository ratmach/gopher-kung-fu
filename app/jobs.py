from __future__ import annotations

import asyncio
import uuid
from collections import defaultdict
from typing import Any

from app.models import Job, JobKind, utcnow
from app.paths import PROJECTS_DIR, project_dir


class JobCancelled(Exception):
    pass


def job_path(slug: str, job_id: str) -> Path:
    return project_dir(slug) / "jobs" / f"{job_id}.json"


class JobHub:
    def __init__(self) -> None:
        self.jobs: dict[str, Job] = {}
        self._queues: dict[str, list[asyncio.Queue[dict]]] = defaultdict(list)
        self._tasks: dict[str, asyncio.Task[Any]] = {}
        self._procs: dict[str, Any] = {}
        self._lock = asyncio.Lock()

    def get(self, job_id: str) -> Job:
        if job_id not in self.jobs:
            raise KeyError(job_id)
        return self.jobs[job_id]

    def list_for(self, slug: str, kind: JobKind | None = None) -> list[Job]:
        matches = [
            job
            for job in self.jobs.values()
            if job.project_slug == slug and (kind is None or job.kind == kind)
        ]
        matches.sort(key=lambda job: job.created_at, reverse=True)
        return matches

    def latest_for(self, slug: str, kind: JobKind | None = None) -> Job | None:
        matches = self.list_for(slug, kind)
        return matches[0] if matches else None

    def running_for(self, slug: str, kind: JobKind) -> Job | None:
        for job in self.list_for(slug, kind):
            if job.is_active():
                return job
        return None

    async def create(self, slug: str, kind: JobKind, revert_status: str | None = None) -> Job:
        job = Job(id=uuid.uuid4().hex[:12], project_slug=slug, kind=kind, revert_status=revert_status)
        async with self._lock:
            self.jobs[job.id] = job
        self._persist(job)
        return job

    def mark_running(self, job_id: str) -> None:
        job = self.jobs[job_id]
        job.status = "running"
        job.updated_at = utcnow()
        self._persist(job)

    def track_task(self, job_id: str, task: asyncio.Task[Any]) -> None:
        self._tasks[job_id] = task

    def attach_process(self, job_id: str, proc: Any) -> None:
        self._procs[job_id] = proc

    def check(self, job_id: str) -> None:
        job = self.jobs.get(job_id)
        if job is not None and job.status == "cancelled":
            raise JobCancelled()

    async def log(self, job_id: str, message: str, progress: float | None = None) -> None:
        async with self._lock:
            job = self.jobs[job_id]
            job.log.append(message)
            if len(job.log) > 4000:
                job.log = job.log[-3000:]
            job.updated_at = utcnow()
            if progress is not None:
                job.progress = max(0.0, min(1.0, progress))
            self._persist(job)
            snapshot = {"type": "log", "message": message, "progress": job.progress, "status": job.status}
        await self._broadcast(job_id, snapshot)

    async def finish(self, job_id: str, status: str, error: str | None = None) -> None:
        job = self.jobs[job_id]
        if job.status == "cancelled" and status != "cancelled":
            status = "cancelled"
        job.status = status  # type: ignore[assignment]
        job.error = error
        job.progress = 1.0 if status == "done" else job.progress
        job.updated_at = utcnow()
        self._persist(job)
        event = "done" if status == "done" else status
        await self._broadcast(
            job_id,
            {"type": event, "status": status, "error": error, "progress": job.progress},
        )
        self._tasks.pop(job_id, None)
        self._procs.pop(job_id, None)

    async def cancel(self, job_id: str) -> Job:
        job = self.get(job_id)
        if not job.is_active():
            return job
        job.status = "cancelled"
        job.error = "cancelled"
        job.updated_at = utcnow()
        job.log.append("Cancel requested — stopping this run.")
        self._persist(job)
        self._kill_process(job_id)
        task = self._tasks.get(job_id)
        if task is not None and not task.done():
            task.cancel()
        await self._broadcast(
            job_id,
            {"type": "log", "message": "Cancel requested — stopping this run.", "progress": job.progress, "status": job.status},
        )
        await self.finish(job_id, "cancelled", "cancelled")
        return job

    def restore(self) -> None:
        if not PROJECTS_DIR.exists():
            return
        for folder in PROJECTS_DIR.iterdir():
            jobs_dir = folder / "jobs"
            if not jobs_dir.is_dir():
                continue
            for path in jobs_dir.glob("*.json"):
                try:
                    job = Job.model_validate_json(path.read_text(encoding="utf-8-sig"))
                except (OSError, ValueError):
                    continue
                if job.is_active():
                    job.status = "error"
                    job.error = "interrupted: factory process restarted"
                    job.log.append(job.error)
                    job.updated_at = utcnow()
                    path.write_text(job.model_dump_json(indent=2), encoding="utf-8")
                self.jobs[job.id] = job

    def subscribe(self) -> asyncio.Queue[dict]:
        queue: asyncio.Queue[dict] = asyncio.Queue()
        return queue

    def attach(self, job_id: str, queue: asyncio.Queue[dict]) -> None:
        self._queues[job_id].append(queue)
        job = self.jobs[job_id]
        for line in job.log:
            queue.put_nowait({"type": "log", "message": line, "progress": job.progress, "status": job.status})
        if not job.is_active():
            queue.put_nowait(
                {
                    "type": "done" if job.status == "done" else job.status,
                    "status": job.status,
                    "error": job.error,
                    "progress": job.progress,
                }
            )

    def detach(self, job_id: str, queue: asyncio.Queue[dict]) -> None:
        queues = self._queues.get(job_id, [])
        if queue in queues:
            queues.remove(queue)

    async def _broadcast(self, job_id: str, payload: dict) -> None:
        for queue in list(self._queues.get(job_id, [])):
            await queue.put(payload)

    def _persist(self, job: Job) -> None:
        path = job_path(job.project_slug, job.id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(job.model_dump_json(indent=2), encoding="utf-8")

    def _kill_process(self, job_id: str) -> None:
        proc = self._procs.get(job_id)
        if proc is None or proc.poll() is not None:
            return
        proc.terminate()
        try:
            proc.wait(timeout=8)
        except Exception:
            proc.kill()


hub = JobHub()
