from __future__ import annotations

import asyncio

from app.jobs import JobCancelled, hub
from app.models import Project
from app.paths import cartridge_dir
from app.pipeline.curriculum import generate_curriculum
from app.pipeline.distill import distill_project
from app.pipeline.export import latest_run_id, spawn_export, write_card
from app.pipeline.train import spawn_train, write_train_config
from app.secrets import SecretStore
from app.store import ProjectStore
from app.teachers.client import TeacherClient, TeacherError


async def _teacher(project: Project, secrets: SecretStore) -> TeacherClient:
    key = secrets.resolve_teacher_key(project.slug, project.teacher_preset)
    if not key:
        raise TeacherError("no API key stored for this specialist or teacher preset")
    return TeacherClient(project, key)


def _restore_status(slug: str, store: ProjectStore, job_id: str) -> None:
    job = hub.get(job_id)
    project = store.get(slug)
    if job.revert_status:
        project.status = job.revert_status  # type: ignore[assignment]
        project.error = None
        store.save(project)


async def _fail(job_id: str, slug: str, store: ProjectStore, exc: BaseException) -> None:
    if isinstance(exc, (JobCancelled, asyncio.CancelledError)):
        _restore_status(slug, store, job_id)
        try:
            await asyncio.shield(hub.log(job_id, "Cancelled. The run stopped; you can start it again."))
            await asyncio.shield(hub.finish(job_id, "cancelled", "cancelled"))
        except Exception:
            pass
        return
    project = store.get(slug)
    project.status = "error"
    project.error = str(exc)
    store.save(project)
    await hub.finish(job_id, "error", str(exc))


async def run_curriculum(job_id: str, slug: str, store: ProjectStore, secrets: SecretStore) -> None:
    project = store.get(slug)
    try:
        hub.check(job_id)
        project.error = None
        store.save(project)
        await hub.log(job_id, "Starting curriculum generation")
        client = await _teacher(project, secrets)
        curriculum = await generate_curriculum(project, client, hub, job_id)
        hub.check(job_id)
        store.save_curriculum(slug, curriculum)
        project.status = "curriculum"
        project.error = None
        store.save(project)
        await hub.finish(job_id, "done")
    except (Exception, asyncio.CancelledError) as exc:
        await _fail(job_id, slug, store, exc)


async def run_distill(job_id: str, slug: str, store: ProjectStore, secrets: SecretStore) -> None:
    project = store.get(slug)
    try:
        hub.check(job_id)
        project.status = "distilling"
        project.error = None
        store.save(project)
        client = await _teacher(project, secrets)
        curriculum = store.load_curriculum(slug)
        if not curriculum.items:
            raise TeacherError("generate and review a curriculum first")
        train_n, eval_n = await distill_project(project, curriculum, client, store, hub, job_id)
        hub.check(job_id)
        project = store.get(slug)
        project.distill.train_count = train_n
        project.distill.eval_count = eval_n
        project.status = "ready_to_train"
        project.error = None
        store.save(project)
        await hub.finish(job_id, "done")
    except (Exception, asyncio.CancelledError) as exc:
        await _fail(job_id, slug, store, exc)


async def _pump_process(job_id: str, proc) -> int:
    assert proc.stdout is not None
    loop = asyncio.get_running_loop()
    while True:
        hub.check(job_id)
        line = await loop.run_in_executor(None, proc.stdout.readline)
        if line == "":
            break
        text = line.rstrip()
        if text:
            await hub.log(job_id, text)
    return await loop.run_in_executor(None, proc.wait)


async def run_train(job_id: str, slug: str, store: ProjectStore) -> None:
    project = store.get(slug)
    try:
        hub.check(job_id)
        train_path = store.train_jsonl(slug)
        if not train_path.exists() or train_path.stat().st_size == 0:
            raise RuntimeError("no synthetic train.jsonl — distill first")
        project.status = "training"
        project.error = None
        store.save(project)
        config_path = write_train_config(project, job_id)
        await hub.log(job_id, f"Launching Unsloth QLoRA worker ({config_path})")
        proc = spawn_train(config_path)
        hub.attach_process(job_id, proc)
        code = await _pump_process(job_id, proc)
        hub.check(job_id)
        if code != 0:
            raise RuntimeError(f"train worker exited with {code}")
        project = store.get(slug)
        project.status = "trained"
        project.last_run_id = job_id
        store.save(project)
        await hub.finish(job_id, "done")
    except (Exception, asyncio.CancelledError) as exc:
        await _fail(job_id, slug, store, exc)


async def run_export(job_id: str, slug: str, store: ProjectStore) -> None:
    project = store.get(slug)
    try:
        hub.check(job_id)
        run_id = project.last_run_id or latest_run_id(slug)
        if not run_id:
            raise RuntimeError("no finished training run to export")
        project.status = "exporting"
        project.error = None
        store.save(project)
        await hub.log(job_id, f"Exporting run {run_id} to GGUF Q4_K_M")
        proc = spawn_export(project, run_id)
        hub.attach_process(job_id, proc)
        code = await _pump_process(job_id, proc)
        hub.check(job_id)
        if code != 0:
            raise RuntimeError(f"export worker exited with {code}")
        dest = cartridge_dir(slug)
        gguf_name = f"{slug}.Q4_K_M.gguf"
        write_card(project, dest, gguf_name)
        project = store.get(slug)
        project.status = "exported"
        project.cartridge_path = str(dest / gguf_name)
        store.save(project)
        await hub.log(job_id, f"Cartridge ready at {dest}. Restart or reload the farm server to list it.")
        await hub.finish(job_id, "done")
    except (Exception, asyncio.CancelledError) as exc:
        await _fail(job_id, slug, store, exc)
