from __future__ import annotations

from functools import lru_cache

from app.jobs import hub
from app.secret_store import SecretStore
from app.store import ProjectStore


@lru_cache
def get_store() -> ProjectStore:
    return ProjectStore()


@lru_cache
def get_secrets() -> SecretStore:
    return SecretStore()


def get_hub():
    return hub
