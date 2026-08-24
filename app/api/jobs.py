from __future__ import annotations

import json

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from app.deps import get_hub
from app.jobs import JobHub
from app.models import Job

router = APIRouter(prefix="/jobs", tags=["jobs"])

TERMINAL = {"done", "error", "cancelled"}


@router.get("/{job_id}")
def get_job(job_id: str, hub: JobHub = Depends(get_hub)) -> Job:
    try:
        return hub.get(job_id)
    except KeyError as exc:
        raise HTTPException(404, "job not found") from exc


@router.post("/{job_id}/cancel")
async def cancel_job(job_id: str, hub: JobHub = Depends(get_hub)) -> Job:
    try:
        return await hub.cancel(job_id)
    except KeyError as exc:
        raise HTTPException(404, "job not found") from exc


@router.get("/{job_id}/events")
async def job_events(job_id: str, hub: JobHub = Depends(get_hub)) -> StreamingResponse:
    try:
        hub.get(job_id)
    except KeyError as exc:
        raise HTTPException(404, "job not found") from exc

    queue = hub.subscribe()
    hub.attach(job_id, queue)

    async def gen():
        try:
            while True:
                payload = await queue.get()
                yield f"data: {json.dumps(payload)}\n\n"
                if payload.get("type") in TERMINAL:
                    break
        finally:
            hub.detach(job_id, queue)

    return StreamingResponse(gen(), media_type="text/event-stream")
