from __future__ import annotations

import json
import re
from pathlib import Path

from app.models import Curriculum, Project, utcnow
from app.paths import LIBRARY_DIR, PROJECTS_DIR, ensure_dirs, project_dir

SLUG_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


def slugify(name: str) -> str:
    text = name.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = text.strip("-")
    if not text:
        raise ValueError("name does not produce a valid slug")
    return text


def validate_slug(slug: str) -> str:
    slug = slug.strip().lower()
    if not SLUG_RE.match(slug):
        raise ValueError("slug must be lowercase letters, digits, and hyphens")
    return slug


class ProjectStore:
    def __init__(self) -> None:
        ensure_dirs()

    def list(self) -> list[Project]:
        projects: list[Project] = []
        if not PROJECTS_DIR.exists():
            return projects
        for path in sorted(PROJECTS_DIR.iterdir()):
            manifest = path / "project.json"
            if not manifest.is_file():
                continue
            try:
                projects.append(self._read_project(manifest))
            except (OSError, ValueError) as exc:
                print(f"skip {manifest}: {exc}")
        return projects

    def get(self, slug: str) -> Project:
        path = self._project_path(slug)
        if not path.exists():
            raise FileNotFoundError(slug)
        return self._read_project(path)

    def exists(self, slug: str) -> bool:
        return self._project_path(slug).exists()

    def create(self, project: Project) -> Project:
        folder = project_dir(project.slug)
        if folder.exists():
            raise FileExistsError(project.slug)
        folder.mkdir(parents=True)
        (folder / "synthetic").mkdir(exist_ok=True)
        (folder / "runs").mkdir(exist_ok=True)
        self.save(project)
        return project

    def save(self, project: Project) -> Project:
        project.updated_at = utcnow()
        path = self._project_path(project.slug)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(project.model_dump_json(indent=2), encoding="utf-8")
        return project

    def delete(self, slug: str) -> None:
        folder = project_dir(slug)
        if not folder.exists():
            raise FileNotFoundError(slug)
        import shutil

        shutil.rmtree(folder)

    def curriculum_path(self, slug: str) -> Path:
        return project_dir(slug) / "curriculum.json"

    def load_curriculum(self, slug: str) -> Curriculum:
        path = self.curriculum_path(slug)
        if not path.exists():
            return Curriculum()
        return Curriculum.model_validate_json(path.read_text(encoding="utf-8-sig"))

    def save_curriculum(self, slug: str, curriculum: Curriculum) -> Curriculum:
        path = self.curriculum_path(slug)
        path.write_text(curriculum.model_dump_json(indent=2), encoding="utf-8")
        return curriculum

    def train_jsonl(self, slug: str) -> Path:
        return project_dir(slug) / "synthetic" / "train.jsonl"

    def eval_jsonl(self, slug: str) -> Path:
        return project_dir(slug) / "synthetic" / "eval.jsonl"

    def inbox_jsonl(self, slug: str) -> Path:
        return project_dir(slug) / "synthetic" / "inbox.jsonl"

    def project_topic_jsonl(self, slug: str, topic: str) -> Path:
        from app.pipeline.library import topic_slug

        return project_dir(slug) / "synthetic" / "topics" / f"{topic_slug(topic)}.jsonl"

    def library_topic_jsonl(self, topic: str) -> Path:
        from app.pipeline.library import topic_slug

        return LIBRARY_DIR / topic_slug(topic) / "examples.jsonl"

    def _project_path(self, slug: str) -> Path:
        return project_dir(slug) / "project.json"

    def _read_project(self, path: Path) -> Project:
        return Project.model_validate_json(path.read_text(encoding="utf-8-sig"))
