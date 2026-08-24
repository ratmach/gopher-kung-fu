from fastapi import APIRouter

from app.api.jobs import router as jobs_router
from app.api.meta import router as meta_router
from app.api.projects import router as projects_router

api_router = APIRouter()
api_router.include_router(meta_router)
api_router.include_router(projects_router)
api_router.include_router(jobs_router)
