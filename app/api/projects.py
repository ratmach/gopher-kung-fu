from __future__ import annotations

import asyncio

from fastapi import APIRouter, Body, Depends, HTTPException

from app.deps import get_hub, get_secrets, get_store
from app.jobs import JobHub
from app.models import (
    CreateProjectIn,
    Curriculum,
    DistillSettings,
    GenerateCurriculumIn,
    Job,
    PatchProjectIn,
    Project,
    StoreSecretIn,
    TopicRef,
    TrainSettings,
)
from app.pipeline.curriculum import clamp_items_per_topic
from app.pipeline.distill import planned_count
from app.runners import run_curriculum, run_distill, run_export, run_train
from app.secrets import SecretStore
from app.store import ProjectStore, slugify, validate_slug
from app.teachers.client import teacher_supports_batch
from app.teachers.presets import preset_by_id

router = APIRouter(prefix="/projects", tags=["projects"])


def _project_out(
    project: Project,
    secrets: SecretStore,
    store: ProjectStore,
    hub: JobHub | None = None,
) -> dict:
    curriculum = store.load_curriculum(project.slug)
    jobs = [job.summary() for job in hub.list_for(project.slug)] if hub else []
    return {
        **project.model_dump(),
        "has_api_key": secrets.has_teacher_key(project.slug, project.teacher_preset),
        "curriculum_count": len(curriculum.items),
        "planned_examples": planned_count(project, curriculum) if curriculum.items else 0,
        "batch_available": teacher_supports_batch(project),
        "jobs": jobs,
    }


@router.get("")
def list_projects(
    store: ProjectStore = Depends(get_store),
    secrets: SecretStore = Depends(get_secrets),
    hub: JobHub = Depends(get_hub),
) -> dict:
    items = [_project_out(project, secrets, store, hub) for project in store.list()]
    return {"projects": items}


@router.post("", status_code=201)
def create_project(
    body: CreateProjectIn,
    store: ProjectStore = Depends(get_store),
    secrets: SecretStore = Depends(get_secrets),
) -> dict:
    try:
        slug = validate_slug(body.slug) if body.slug else slugify(body.name)
        preset_by_id(body.teacher_preset)
    except (ValueError, KeyError) as exc:
        raise HTTPException(400, str(exc)) from exc
    if store.exists(slug):
        raise HTTPException(409, f"specialist '{slug}' already exists")
    preset = preset_by_id(body.teacher_preset)
    project = Project(
        slug=slug,
        name=body.name.strip(),
        base_model=body.base_model,
        teacher_preset=body.teacher_preset,
        teacher_model=body.teacher_model or preset["model"],
        teacher_base_url=body.teacher_base_url or preset["base_url"],
    )
    store.create(project)
    if body.api_key:
        secrets.put(f"project:{slug}", body.api_key)
        secrets.put(f"preset:{body.teacher_preset}", body.api_key)
    return _project_out(project, secrets, store, get_hub())


@router.get("/{slug}")
def get_project(
    slug: str,
    store: ProjectStore = Depends(get_store),
    secrets: SecretStore = Depends(get_secrets),
    hub: JobHub = Depends(get_hub),
) -> dict:
    try:
        project = store.get(slug)
    except FileNotFoundError as exc:
        raise HTTPException(404, "specialist not found") from exc
    return _project_out(project, secrets, store, hub)


@router.patch("/{slug}")
def patch_project(
    slug: str,
    body: PatchProjectIn,
    store: ProjectStore = Depends(get_store),
    secrets: SecretStore = Depends(get_secrets),
) -> dict:
    try:
        project = store.get(slug)
    except FileNotFoundError as exc:
        raise HTTPException(404, "specialist not found") from exc
    fields = body.model_fields_set
    api_key = body.api_key if "api_key" in fields else None
    scalars = body.model_dump(exclude_unset=True, exclude={"api_key", "topics", "distill", "train"})
    for key, value in scalars.items():
        setattr(project, key, value)
    if "topics" in fields and body.topics is not None:
        project.topics = [TopicRef.model_validate(item) for item in body.topics]
    if "distill" in fields and body.distill is not None:
        project.distill = DistillSettings.model_validate(body.distill)
    if "train" in fields and body.train is not None:
        project.train = TrainSettings.model_validate(body.train)
    if project.teacher_preset:
        try:
            preset = preset_by_id(project.teacher_preset)
        except KeyError as exc:
            raise HTTPException(400, str(exc)) from exc
        if not project.teacher_model:
            project.teacher_model = preset["model"]
        if not project.teacher_base_url:
            project.teacher_base_url = preset["base_url"]
    store.save(project)
    if api_key:
        secrets.put(f"project:{slug}", api_key)
        secrets.put(f"preset:{project.teacher_preset}", api_key)
    return _project_out(project, secrets, store, get_hub())


@router.get("/{slug}/jobs")
def project_jobs(slug: str, store: ProjectStore = Depends(get_store), hub: JobHub = Depends(get_hub)) -> dict:
    if not store.exists(slug):
        raise HTTPException(404, "specialist not found")
    return {"jobs": [job.summary() for job in hub.list_for(slug)]}


@router.delete("/{slug}", status_code=204)
def delete_project(slug: str, store: ProjectStore = Depends(get_store)) -> None:
    try:
        store.delete(slug)
    except FileNotFoundError as exc:
        raise HTTPException(404, "specialist not found") from exc


@router.get("/{slug}/curriculum")
def get_curriculum(slug: str, store: ProjectStore = Depends(get_store)) -> Curriculum:
    if not store.exists(slug):
        raise HTTPException(404, "specialist not found")
    return store.load_curriculum(slug)


@router.put("/{slug}/curriculum")
def put_curriculum(
    slug: str,
    body: Curriculum,
    store: ProjectStore = Depends(get_store),
) -> Curriculum:
    try:
        project = store.get(slug)
    except FileNotFoundError as exc:
        raise HTTPException(404, "specialist not found") from exc
    store.save_curriculum(slug, body)
    if body.items and project.status == "draft":
        project.status = "curriculum"
        store.save(project)
    return body


@router.post("/{slug}/secret")
def store_secret(
    slug: str,
    body: StoreSecretIn,
    store: ProjectStore = Depends(get_store),
    secrets: SecretStore = Depends(get_secrets),
) -> dict:
    try:
        project = store.get(slug)
    except FileNotFoundError as exc:
        raise HTTPException(404, "specialist not found") from exc
    secrets.put(f"project:{slug}", body.api_key)
    secrets.put(f"preset:{project.teacher_preset}", body.api_key)
    return {"ok": True}


async def _start_job(
    slug: str,
    kind: str,
    store: ProjectStore,
    hub: JobHub,
    coro,
) -> Job:
    if not store.exists(slug):
        raise HTTPException(404, "specialist not found")
    existing = hub.running_for(slug, kind)  # type: ignore[arg-type]
    if existing:
        return existing
    project = store.get(slug)
    job = await hub.create(slug, kind, revert_status=project.status)  # type: ignore[arg-type]
    hub.mark_running(job.id)
    task = asyncio.create_task(coro(job.id, slug))
    hub.track_task(job.id, task)
    return job


@router.post("/{slug}/curriculum/generate")
async def generate_curriculum_job(
    slug: str,
    body: GenerateCurriculumIn = Body(default_factory=GenerateCurriculumIn),
    store: ProjectStore = Depends(get_store),
    secrets: SecretStore = Depends(get_secrets),
    hub: JobHub = Depends(get_hub),
) -> Job:
    try:
        project = store.get(slug)
    except FileNotFoundError as exc:
        raise HTTPException(404, "specialist not found") from exc
    if not project.topics:
        raise HTTPException(400, "select at least one topic")
    if body.items_per_topic is not None:
        project.items_per_topic = clamp_items_per_topic(body.items_per_topic)
        store.save(project)
    return await _start_job(
        slug,
        "curriculum",
        store,
        hub,
        lambda job_id, s: run_curriculum(job_id, s, store, secrets),
    )


@router.post("/{slug}/distill")
async def distill_job(
    slug: str,
    store: ProjectStore = Depends(get_store),
    secrets: SecretStore = Depends(get_secrets),
    hub: JobHub = Depends(get_hub),
) -> Job:
    return await _start_job(
        slug,
        "distill",
        store,
        hub,
        lambda job_id, s: run_distill(job_id, s, store, secrets),
    )


@router.post("/{slug}/train")
async def train_job(
    slug: str,
    store: ProjectStore = Depends(get_store),
    hub: JobHub = Depends(get_hub),
) -> Job:
    return await _start_job(
        slug,
        "train",
        store,
        hub,
        lambda job_id, s: run_train(job_id, s, store),
    )


@router.post("/{slug}/export")
async def export_job(
    slug: str,
    store: ProjectStore = Depends(get_store),
    hub: JobHub = Depends(get_hub),
) -> Job:
    return await _start_job(
        slug,
        "export",
        store,
        hub,
        lambda job_id, s: run_export(job_id, s, store),
    )
