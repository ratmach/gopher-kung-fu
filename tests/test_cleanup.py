from pathlib import Path

from app.pipeline.cleanup import (
    cleanup_after_export,
    cleanup_after_train,
    format_bytes,
    path_bytes,
    remove_path,
)


def _write(path: Path, size: int = 32) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x" * size)


def test_format_bytes():
    assert format_bytes(0) == "0 B"
    assert format_bytes(512) == "512 B"
    assert format_bytes(1536).endswith("KB")
    assert format_bytes(5 * 1024 * 1024).endswith("MB")


def test_cleanup_after_train_drops_trainer_keeps_adapter(tmp_path: Path):
    run = tmp_path / "runs" / "abc"
    _write(run / "adapter" / "adapter_model.safetensors", 64)
    _write(run / "adapter" / "trainer" / "checkpoint-1" / "optimizer.pt", 128)
    _write(run / "merged" / "model.safetensors", 256)
    removed, freed = cleanup_after_train(run)
    assert freed == 128
    assert any("trainer" in item for item in removed)
    assert (run / "adapter" / "adapter_model.safetensors").is_file()
    assert (run / "merged" / "model.safetensors").is_file()
    assert not (run / "adapter" / "trainer").exists()


def test_cleanup_after_export_drops_merged_and_old_runs(tmp_path: Path, monkeypatch):
    from app.pipeline import cleanup as cleanup_mod

    slug = "gopher-go"
    monkeypatch.setattr(cleanup_mod, "project_dir", lambda name: tmp_path / "projects" / name)
    monkeypatch.setattr(cleanup_mod, "cartridge_dir", lambda name: tmp_path / "cartridges" / name)

    keep = tmp_path / "projects" / slug / "runs" / "keep"
    old = tmp_path / "projects" / slug / "runs" / "old"
    cart = tmp_path / "cartridges" / slug
    _write(keep / "adapter" / "adapter_model.safetensors", 40)
    _write(keep / "adapter" / "trainer" / "optimizer.pt", 80)
    _write(keep / "merged" / "model.safetensors", 200)
    _write(old / "merged" / "model.safetensors", 300)
    _write(cart / "gopher-go.Q4_K_M.gguf", 50)
    _write(cart / "gopher-go.f16.gguf", 90)
    _write(cart / "_gguf_tmp" / "scratch.bin", 20)

    removed, freed = cleanup_after_export(slug, "keep")
    assert freed == 80 + 200 + 300 + 90 + 20
    assert not (keep / "merged").exists()
    assert not (keep / "adapter" / "trainer").exists()
    assert (keep / "adapter" / "adapter_model.safetensors").is_file()
    assert not old.exists()
    assert (cart / "gopher-go.Q4_K_M.gguf").is_file()
    assert not (cart / "gopher-go.f16.gguf").exists()
    assert not (cart / "_gguf_tmp").exists()
    assert removed


def test_remove_missing_is_zero(tmp_path: Path):
    assert remove_path(tmp_path / "nope") == 0
    assert path_bytes(tmp_path / "nope") == 0
