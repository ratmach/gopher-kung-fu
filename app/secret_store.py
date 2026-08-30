from __future__ import annotations

import json
from threading import Lock

from app.paths import SECRETS_PATH, ensure_dirs


class SecretStore:
    def __init__(self) -> None:
        self._lock = Lock()
        ensure_dirs()

    def _load(self) -> dict[str, str]:
        if not SECRETS_PATH.exists():
            return {}
        data = json.loads(SECRETS_PATH.read_text(encoding="utf-8-sig"))
        return dict(data.get("keys", {}))

    def _save(self, keys: dict[str, str]) -> None:
        SECRETS_PATH.write_text(
            json.dumps({"keys": keys}, indent=2),
            encoding="utf-8",
        )

    def put(self, key: str, value: str) -> None:
        value = value.strip()
        if not value:
            return
        with self._lock:
            keys = self._load()
            keys[key] = value
            self._save(keys)

    def get(self, key: str) -> str | None:
        with self._lock:
            return self._load().get(key)

    def has(self, key: str) -> bool:
        return bool(self.get(key))

    def resolve_teacher_key(self, slug: str, preset_id: str) -> str | None:
        return self.get(f"project:{slug}") or self.get(f"preset:{preset_id}")

    def has_teacher_key(self, slug: str, preset_id: str) -> bool:
        return bool(self.resolve_teacher_key(slug, preset_id))
