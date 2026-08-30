from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

ProjectStatus = Literal[
    "draft",
    "curriculum",
    "distilling",
    "ready_to_train",
    "training",
    "trained",
    "exporting",
    "exported",
    "error",
]

Skill = Literal["write", "review", "debug", "refactor", "idiom"]
Difficulty = Literal["easy", "medium", "hard"]
JobKind = Literal["curriculum", "distill", "train", "export"]
JobStatus = Literal["queued", "running", "done", "error", "cancelled"]
BaseModelId = Literal[
    "qwen3-1.7b",
    "qwen2.5-coder-3b",
    "qwen2.5-coder-7b",
    "qwen3-4b",
    "deepseek-coder-v2",
]


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


class TopicRef(BaseModel):
    id: str
    label: str
    custom: bool = False


class DistillSettings(BaseModel):
    examples_per_topic: int = 100
    train_count: int = 0
    eval_count: int = 0
    use_batch: bool = False


class TrainSettings(BaseModel):
    lora_r: int = 16
    lora_alpha: int = 16
    epochs: float = 2.0
    seq_len: int = 2048
    batch_size: int = 1
    grad_accum: int = 8
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.05
    eval_steps: int = 0
    train_on_responses_only: bool = True


class Project(BaseModel):
    model_config = ConfigDict(validate_assignment=True)

    slug: str
    name: str
    base_model: BaseModelId = "qwen3-1.7b"
    teacher_preset: str = "deepseek"
    teacher_model: str = ""
    teacher_base_url: str = ""
    topics: list[TopicRef] = Field(default_factory=list)
    status: ProjectStatus = "draft"
    error: str | None = None
    created_at: str = Field(default_factory=utcnow)
    updated_at: str = Field(default_factory=utcnow)
    distill: DistillSettings = Field(default_factory=DistillSettings)
    train: TrainSettings = Field(default_factory=TrainSettings)
    items_per_topic: int = 12
    last_run_id: str | None = None
    cartridge_path: str | None = None
    allowed_imports: list[str] = Field(
        default_factory=lambda: ["encoding/csv", "encoding/json"]
    )
    forbidden_imports: list[str] = Field(default_factory=lambda: ["C"])


class CurriculumItem(BaseModel):
    id: str
    topic: str
    subtopic: str
    skill: Skill
    difficulty: Difficulty = "medium"
    notes: str = ""


class Curriculum(BaseModel):
    items: list[CurriculumItem] = Field(default_factory=list)
    generated_at: str | None = None


class ShareGPTTurn(BaseModel):
    role: Literal["human", "gpt"]
    value: str

    def to_unsloth(self) -> dict:
        return {"from": self.role, "value": self.value}


class ShareGPTExample(BaseModel):
    conversations: list[ShareGPTTurn]
    meta: dict = Field(default_factory=dict)


class JobEvent(BaseModel):
    ts: str = Field(default_factory=utcnow)
    message: str
    progress: float | None = None


class Job(BaseModel):
    id: str
    project_slug: str
    kind: JobKind
    status: JobStatus = "queued"
    progress: float = 0.0
    error: str | None = None
    revert_status: str | None = None
    created_at: str = Field(default_factory=utcnow)
    updated_at: str = Field(default_factory=utcnow)
    log: list[str] = Field(default_factory=list)

    def is_active(self) -> bool:
        return self.status in {"queued", "running"}

    def summary(self) -> dict:
        return self.model_dump(exclude={"log"})


class CreateProjectIn(BaseModel):
    name: str
    slug: str | None = None
    base_model: BaseModelId = "qwen3-1.7b"
    teacher_preset: str = "deepseek"
    teacher_model: str = ""
    teacher_base_url: str = ""
    api_key: str | None = None


class PatchProjectIn(BaseModel):
    name: str | None = None
    base_model: BaseModelId | None = None
    teacher_preset: str | None = None
    teacher_model: str | None = None
    teacher_base_url: str | None = None
    topics: list[TopicRef] | None = None
    distill: DistillSettings | None = None
    train: TrainSettings | None = None
    items_per_topic: int | None = None
    allowed_imports: list[str] | None = None
    api_key: str | None = None


class GenerateCurriculumIn(BaseModel):
    items_per_topic: int | None = None


class StoreSecretIn(BaseModel):
    api_key: str
    scope: str | None = None
