from __future__ import annotations

from functools import lru_cache
from typing import Any

import yaml

from app.paths import CATALOG_PATH


@lru_cache
def load_catalog() -> dict[str, Any]:
    with CATALOG_PATH.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def flatten_topics() -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for category in load_catalog().get("categories", []):
        for topic in category.get("topics", []):
            out.append(
                {
                    "id": topic["id"],
                    "label": topic["label"],
                    "category": category["id"],
                    "category_label": category["label"],
                }
            )
    return out
