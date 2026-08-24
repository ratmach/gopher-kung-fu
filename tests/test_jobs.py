import asyncio

from app.jobs import JobCancelled, JobHub, job_path
from app.models import Job


def test_job_persist_restore_and_orphan(tmp_path, monkeypatch):
    monkeypatch.setattr("app.jobs.PROJECTS_DIR", tmp_path)
    monkeypatch.setattr("app.jobs.project_dir", lambda slug: tmp_path / slug)

    hub = JobHub()
    job = Job(id="abc123abc123", project_slug="gopher", kind="distill", status="running")
    hub.jobs[job.id] = job
    hub._persist(job)
    assert job_path("gopher", job.id).is_file()

    other = JobHub()
    monkeypatch.setattr("app.jobs.PROJECTS_DIR", tmp_path)
    other.restore()
    restored = other.get(job.id)
    assert restored.status == "error"
    assert restored.error and "restarted" in restored.error


def test_check_cancelled():
    hub = JobHub()
    hub.jobs["x"] = Job(id="x", project_slug="gopher", kind="distill", status="cancelled")
    try:
        hub.check("x")
        assert False, "expected JobCancelled"
    except JobCancelled:
        pass


def test_cancel_inactive_is_noop():
    async def run():
        hub = JobHub()
        job = Job(id="done1", project_slug="gopher", kind="train", status="done")
        hub.jobs[job.id] = job
        out = await hub.cancel(job.id)
        assert out.status == "done"

    asyncio.run(run())
