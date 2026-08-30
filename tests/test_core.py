import os
from pathlib import Path

from app.models import Curriculum, CurriculumItem, Project, ShareGPTExample, ShareGPTTurn
from app.pipeline.distill import planned_count
from app.pipeline.filters import keep_example
from app.pipeline.gpu import require_cuda
from app.store import ProjectStore, slugify, validate_slug
from app.teachers.client import parse_json_payload, strip_think


def test_base_model_catalog():
    from typing import get_args

    from app.models import BaseModelId
    from app.teachers.presets import BASE_MODELS, base_model_spec, response_mask_parts, train_defaults_for

    assert set(BASE_MODELS) == set(get_args(BaseModelId))
    assert "ministral-3b" not in BASE_MODELS
    assert BASE_MODELS["qwen3-1.7b"]["default"] is True
    assert base_model_spec("qwen2.5-coder-3b")["train_id"].endswith("Qwen2.5-Coder-3B-Instruct-bnb-4bit")
    assert base_model_spec("deepseek-coder-v2")["trust_remote_code"] is True
    assert train_defaults_for("qwen2.5-coder-7b") == {"seq_len": 4096, "learning_rate": 1e-4}
    assert train_defaults_for("qwen3-4b") == {"seq_len": 4096, "learning_rate": 1e-4}
    assert train_defaults_for("qwen2.5-coder-3b") == {"seq_len": 4096, "learning_rate": 1e-4}
    assert train_defaults_for("deepseek-coder-v2") == {"seq_len": 4096, "learning_rate": 1e-4}
    assert train_defaults_for("qwen3-1.7b") == {}
    assert response_mask_parts("deepseek-coder-v2")["instruction_part"] == "User:"
    assert response_mask_parts("qwen3-4b")["instruction_part"].startswith("<|im_start|>")


def test_slugify():
    assert slugify("Gopher Kungfu") == "gopher-kungfu"
    assert validate_slug("sql-specialist") == "sql-specialist"


def test_keep_example_requires_code_for_write():
    bad = ShareGPTExample(
        conversations=[
            ShareGPTTurn(role="human", value="Write a mutex"),
            ShareGPTTurn(role="gpt", value="Just use a lock."),
        ],
        meta={"skill": "write"},
    )
    good = ShareGPTExample(
        conversations=[
            ShareGPTTurn(role="human", value="Write a mutex"),
            ShareGPTTurn(role="gpt", value="Use:\n```go\nvar mu sync.Mutex\n```\n"),
        ],
        meta={"skill": "write"},
    )
    assert keep_example(bad) is False
    assert keep_example(good) is True
    essay = ShareGPTExample(
        conversations=[
            ShareGPTTurn(role="human", value="Write a mutex"),
            ShareGPTTurn(
                role="gpt",
                value=(
                    "First consider the architecture, packages, and tradeoffs. "
                    * 20
                    + "\n```go\nvar mu sync.Mutex\n```\n"
                ),
            ),
        ],
        meta={"skill": "write"},
    )
    assert keep_example(essay) is False


def test_distill_debug_prompt_asks_verbatim_gc():
    from app.pipeline.distill import SYSTEM, _user_prompt

    item = CurriculumItem(
        id="for-else",
        topic="Go",
        subtopic="for-else-not-in-go",
        skill="debug",
        difficulty="easy",
        notes="unexpected keyword else",
    )
    project = Project(slug="gopher", name="Gopher")
    prompt = _user_prompt(project, item, 2)
    assert "verbatim" in prompt
    assert "for/else" in prompt
    assert "member list" in prompt
    assert "unexpected keyword else" in prompt
    assert "cannot convert 0" in prompt
    assert "return nil" in prompt
    assert "declared and not used" in prompt
    assert "never `_ = buf`" in prompt
    assert "[]byte(string(r))" in prompt
    assert "map[string][]string" in prompt
    assert "verbatim `gc`" in SYSTEM
    assert "for/else" in SYSTEM
    assert "cannot convert 0" in SYSTEM
    assert "declared and not used" in SYSTEM
    assert "never `_ = buf`" in SYSTEM
    assert "[]byte(string(r))" in SYSTEM
    assert "map[string][]string" in SYSTEM
    assert "member list" in SYSTEM


def test_parse_json_and_think():
    raw = "<think>secret</think>\n```json\n{\"items\":[1]}\n```"
    assert strip_think(raw).startswith("```")
    assert parse_json_payload(raw) == {"items": [1]}


def test_parse_json_ignores_go_fence():
    raw = (
        'Here you go\n```go\npackage main\nfunc main() {}\n```\n'
        '{"examples":[{"human":"design a closer","gpt":"```go\\ntype Closer interface { Close() error }\\n```"}]}'
    )
    assert parse_json_payload(raw)["examples"][0]["human"] == "design a closer"


def test_parse_single_quoted_teacher_json():
    raw = "{'examples': [{'human': 'design a closer', 'gpt': 'type Closer interface {}'}]}"
    assert parse_json_payload(raw)["examples"][0]["human"] == "design a closer"
    wrapped = "Sure.\n{'items': [{'id': 'go-1', 'topic': 'Go', 'subtopic': 'ifaces', 'skill': 'write'}]}\n"
    assert parse_json_payload(wrapped)["items"][0]["id"] == "go-1"


def test_parse_unquoted_js_keys():
    raw = '{examples:[{human:"design a closer",gpt:"type Closer interface {}"}]}'
    assert parse_json_payload(raw)["examples"][0]["human"] == "design a closer"
    assert parse_json_payload("{items:[{id:'go-1',topic:'Go',subtopic:'ifaces',skill:'write'}]}")["items"][0]["id"] == "go-1"


def test_distill_assignment_and_planned_count():
    project = Project(slug="gopher-kungfu1", name="Gopher")
    project.distill = {"examples_per_topic": 40, "train_count": 0, "eval_count": 0}
    assert project.distill.examples_per_topic == 40
    assert project.distill.use_batch is False
    raw = Project.model_construct(
        slug="raw",
        name="Raw",
        distill={"examples_per_topic": 12, "train_count": 0, "eval_count": 0},
    )
    curriculum = Curriculum(
        items=[CurriculumItem(id="go-1", topic="Go", subtopic="chan", skill="write")]
    )
    assert planned_count(raw, curriculum) >= 8


def test_require_cuda_explains_cpu_torch(monkeypatch):
    import app.pipeline.gpu as gpu

    monkeypatch.setattr(gpu, "nvidia_gpu_name", lambda: "NVIDIA GeForce RTX 5070")
    monkeypatch.setattr(gpu, "probe_torch", lambda: {
        "torch": "2.11.0+cpu",
        "cuda_built": None,
        "cuda_available": False,
        "device_name": None,
        "host_gpu": "NVIDIA GeForce RTX 5070",
        "hint": "PyTorch is CPU-only, so Unsloth cannot see the GPU. nvidia-smi sees NVIDIA GeForce RTX 5070.",
    })
    try:
        require_cuda()
        assert False, "expected RuntimeError"
    except RuntimeError as exc:
        assert "CPU-only" in str(exc)
        assert "5070" in str(exc)


def test_load_repo_env_sets_hf_token(tmp_path, monkeypatch):
    from app.paths import load_repo_env

    (tmp_path / ".env").write_text('HF_TOKEN="hf_from_file"\n', encoding="utf-8")
    monkeypatch.setattr("app.paths.ROOT", tmp_path)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    assert load_repo_env() == tmp_path / ".env"
    assert os.environ["HF_TOKEN"] == "hf_from_file"


def test_load_repo_env_keeps_existing_hf_token(tmp_path, monkeypatch):
    from app.paths import load_repo_env

    (tmp_path / ".env").write_text("HF_TOKEN=from_file\n", encoding="utf-8")
    monkeypatch.setattr("app.paths.ROOT", tmp_path)
    monkeypatch.setenv("HF_TOKEN", "from_shell")
    load_repo_env()
    assert os.environ["HF_TOKEN"] == "from_shell"


def test_load_repo_env_skips_blank_hf_token(tmp_path, monkeypatch):
    from app.paths import load_repo_env

    (tmp_path / ".env").write_text("HF_TOKEN=\n", encoding="utf-8")
    monkeypatch.setattr("app.paths.ROOT", tmp_path)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    load_repo_env()
    assert "HF_TOKEN" not in os.environ


def test_unsloth_worker_env_points_at_data_cache():
    from app.paths import UNSLOTH_CACHE
    from app.pipeline.gpu import unsloth_worker_env

    env = unsloth_worker_env({})
    assert env["UNSLOTH_COMPILE_LOCATION"] == str(UNSLOTH_CACHE)
    assert env["PYTHONIOENCODING"] == "utf-8"
    assert env["PYTHONUTF8"] == "1"


def test_train_settings_and_sft_kwargs():
    from app.models import TrainSettings
    from app.pipeline.train_worker import eval_run_plan, sft_config_kwargs

    settings = TrainSettings()
    assert settings.warmup_ratio == 0.05
    assert settings.eval_steps == 0
    assert settings.train_on_responses_only is True
    assert settings.learning_rate == 2e-4

    assert eval_run_plan(has_eval=False, eval_steps=50) == {"eval_strategy": "no"}
    assert eval_run_plan(has_eval=True, eval_steps=0) == {"eval_strategy": "epoch"}
    assert eval_run_plan(has_eval=True, eval_steps=25) == {"eval_strategy": "steps", "eval_steps": 25}

    kwargs = sft_config_kwargs(
        {
            "adapter_dir": "/tmp/adapter",
            "batch_size": 1,
            "grad_accum": 8,
            "epochs": 2,
            "learning_rate": 1e-4,
            "seq_len": 4096,
            "warmup_ratio": 0.1,
            "eval_steps": 25,
        },
        has_eval=True,
    )
    assert kwargs["max_seq_length"] == 4096
    assert kwargs["learning_rate"] == 1e-4
    assert kwargs["warmup_ratio"] == 0.1
    assert kwargs["eval_strategy"] == "steps"
    assert kwargs["eval_steps"] == 25
    assert kwargs["per_device_eval_batch_size"] == 1
    assert kwargs["save_total_limit"] == 1


def test_write_train_config_includes_new_fields(tmp_path, monkeypatch):
    import json

    from app.models import Project
    from app.pipeline import train as train_mod

    monkeypatch.setattr(train_mod, "project_dir", lambda slug: tmp_path / slug)
    monkeypatch.setattr(train_mod, "run_dir", lambda slug, run_id: tmp_path / slug / "runs" / run_id)
    project = Project(
        slug="g7",
        name="G7",
        base_model="qwen2.5-coder-7b",
        train={"seq_len": 4096, "learning_rate": 1e-4},
    )
    path = train_mod.write_train_config(project, "run1")
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["seq_len"] == 4096
    assert data["learning_rate"] == 1e-4
    assert data["warmup_ratio"] == 0.05
    assert data["eval_steps"] == 0
    assert data["train_on_responses_only"] is True
    assert data["instruction_part"] == "<|im_start|>user\n"
    assert data["response_part"] == "<|im_start|>assistant\n"


def test_write_card_allowlists(tmp_path):
    import json

    from app.models import Project, TopicRef
    from app.pipeline.export import write_card

    project = Project(
        slug="gopher-go",
        name="Gopher",
        topics=[TopicRef(id="go", label="Go")],
        allowed_imports=["modernc.org/sqlite"],
    )
    path = write_card(project, tmp_path, "gopher-go.Q4_K_M.gguf")
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["allowed_imports"] == ["modernc.org/sqlite"]
    assert "C" in data["forbidden_imports"]

    dest = tmp_path / "defaults"
    dest.mkdir()
    default_card = write_card(
        Project(slug="gopher-defaults", name="Gopher"),
        dest,
        "gopher-defaults.Q4_K_M.gguf",
    )
    defaults = json.loads(default_card.read_text(encoding="utf-8"))
    assert defaults["allowed_imports"] == ["encoding/csv", "encoding/json"]
    assert data["quant"] == "Q4_K_M"


def test_tag_from_llama_cpp_zip_name():
    from app.pipeline.llama_cpp_tools import _converter_ready, _tag_from_name

    assert _tag_from_name("llama-b10604-bin-win-cpu-x64.zip") == "b10604"
    assert _tag_from_name("nope") is None
    assert _converter_ready(Path(".")) is False


def test_prepend_windows_cmake(tmp_path, monkeypatch):
    from app.pipeline import llama_cpp_tools as tools

    cmake_bin = tmp_path / "CMake" / "bin"
    cmake_bin.mkdir(parents=True)
    monkeypatch.setattr(tools, "windows_cmake_bins", lambda: [cmake_bin])
    env = {"PATH": "old"}
    out = tools.prepend_windows_cmake(env)
    assert str(cmake_bin) in out["PATH"].split(os.pathsep)[0]


def test_read_project_strips_bom(tmp_path):
    project = Project(slug="bom-test", name="BOM")
    path = tmp_path / "project.json"
    path.write_bytes(b"\xef\xbb\xbf" + project.model_dump_json().encode("utf-8"))
    got = ProjectStore()._read_project(path)
    assert got.slug == "bom-test"
