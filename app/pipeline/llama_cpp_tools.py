from __future__ import annotations

import os
import re
import shutil
import sys
import tarfile
import zipfile
from pathlib import Path

import httpx

from app.paths import LLAMA_CPP_TOOLS, ensure_dirs

RELEASES_URL = "https://api.github.com/repos/ggml-org/llama.cpp/releases"
SOURCE_TARBALL = "https://codeload.github.com/ggml-org/llama.cpp/tar.gz/refs/tags/{tag}"
USER_AGENT = "gopher-kung-fu-export"
FALLBACK_WIN_ZIP = (
    "https://github.com/ggml-org/llama.cpp/releases/download/b9561/"
    "llama-b9561-bin-win-cpu-x64.zip"
)
TAG_RE = re.compile(r"(b\d+)")


def windows_cmake_bins() -> list[Path]:
    pf = os.environ.get("ProgramFiles", r"C:\Program Files")
    local = os.environ.get("LOCALAPPDATA", "")
    return [
        Path(pf) / "CMake" / "bin",
        Path(local) / "Programs" / "CMake" / "bin" if local else Path(),
    ]


def prepend_windows_cmake(env: dict[str, str]) -> dict[str, str]:
    """Kitware CMake is often installed but missing from PATH, so Unsloth's which() misses it."""
    extras = [str(path) for path in windows_cmake_bins() if path.is_dir()]
    if not extras:
        return env
    env["PATH"] = os.pathsep.join([*extras, env.get("PATH", os.environ.get("PATH", ""))])
    os.environ["PATH"] = env["PATH"]
    return env


def find_exe(names: list[str], extra_roots: list[Path] | None = None) -> Path | None:
    for name in names:
        found = shutil.which(name)
        if found:
            return Path(found)
    roots: list[Path] = [LLAMA_CPP_TOOLS]
    extra = os.environ.get("LLAMA_CPP_DIR")
    if extra:
        roots.append(Path(extra))
    if extra_roots:
        roots.extend(extra_roots)
    suffixes = [""]
    if sys.platform == "win32":
        suffixes = [".exe", ""]
    for root in roots:
        if not root.exists():
            continue
        for name in names:
            for suffix in suffixes:
                hits = list(root.rglob(name + suffix))
                if hits:
                    return hits[0]
    return None


def _http() -> httpx.Client:
    return httpx.Client(
        headers={"User-Agent": USER_AGENT, "Accept": "application/vnd.github+json"},
        follow_redirects=True,
        timeout=180.0,
    )


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with _http() as client:
        with client.stream("GET", url) as response:
            response.raise_for_status()
            with dest.open("wb") as fh:
                for chunk in response.iter_bytes(1024 * 1024):
                    fh.write(chunk)


def _asset_name() -> str:
    machine = os.environ.get("PROCESSOR_ARCHITECTURE", "").lower()
    if sys.platform == "win32":
        return r"llama-b\d+-bin-win-cpu-x64\.zip"
    if sys.platform == "darwin":
        import platform

        arch = "arm64" if platform.machine().lower() in {"arm64", "aarch64"} else "x64"
        return rf"llama-b\d+-bin-macos-{arch}\.tar\.gz"
    return r"llama-b\d+-bin-ubuntu-x64\.tar\.gz"


def _tag_from_name(name: str) -> str | None:
    match = TAG_RE.search(name)
    return match.group(1) if match else None


def _pick_release_asset() -> tuple[str, str]:
    pattern = re.compile(_asset_name() + r"$")
    try:
        with _http() as client:
            releases = client.get(RELEASES_URL, params={"per_page": 8}).json()
        if isinstance(releases, list):
            for release in releases:
                tag = str(release.get("tag_name") or "")
                for asset in release.get("assets") or []:
                    name = asset.get("name") or ""
                    url = asset.get("browser_download_url") or ""
                    if pattern.match(name) and url:
                        return _tag_from_name(tag) or _tag_from_name(name) or tag, url
    except Exception:
        pass
    if sys.platform == "win32":
        return "b9561", FALLBACK_WIN_ZIP
    raise RuntimeError(
        "Could not find a llama.cpp CPU binary for this OS. "
        "Set LLAMA_CPP_DIR to a folder that contains llama-quantize."
    )


def _extract(archive: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    if archive.name.endswith(".zip"):
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(dest)
        return
    with tarfile.open(archive) as tf:
        tf.extractall(dest)


def _single_extracted_root(extract_dir: Path) -> Path:
    children = [path for path in extract_dir.iterdir() if path.name not in {".", ".."}]
    if len(children) == 1 and children[0].is_dir():
        return children[0]
    return extract_dir


def _converter_ready(root: Path) -> bool:
    script = root / "convert_hf_to_gguf.py"
    conversion = root / "conversion"
    return (
        script.is_file()
        and script.stat().st_size > 1000
        and (conversion / "__init__.py").is_file()
        and (conversion / "base.py").is_file()
    )


def _existing_binary_tag() -> str | None:
    if not LLAMA_CPP_TOOLS.exists():
        return None
    for path in LLAMA_CPP_TOOLS.glob("llama-b*-bin-*"):
        tag = _tag_from_name(path.name)
        if tag:
            return tag
    return None


def _hydrate_converter(tag: str) -> Path:
    """Pull convert_hf_to_gguf.py + conversion/ from the same llama.cpp tag as the binaries."""
    ensure_dirs()
    url = SOURCE_TARBALL.format(tag=tag)
    archive = LLAMA_CPP_TOOLS / f"llama.cpp-{tag}-src.tar.gz"
    print(f"Downloading llama.cpp converter sources {tag} from {url}", flush=True)
    _download(url, archive)
    extract_dir = LLAMA_CPP_TOOLS / f"_src_{tag}"
    if extract_dir.exists():
        shutil.rmtree(extract_dir, ignore_errors=True)
    extract_dir.mkdir(parents=True, exist_ok=True)
    _extract(archive, extract_dir)
    root = _single_extracted_root(extract_dir)
    converter = root / "convert_hf_to_gguf.py"
    conversion = root / "conversion"
    if not converter.is_file() or not conversion.is_dir():
        raise RuntimeError(
            f"llama.cpp {tag} source tarball is missing convert_hf_to_gguf.py / conversion/."
        )
    shutil.copy2(converter, LLAMA_CPP_TOOLS / "convert_hf_to_gguf.py")
    dest_conversion = LLAMA_CPP_TOOLS / "conversion"
    if dest_conversion.exists():
        shutil.rmtree(dest_conversion)
    shutil.copytree(conversion, dest_conversion)
    gguf_py = root / "gguf-py"
    if gguf_py.is_dir():
        dest_gguf = LLAMA_CPP_TOOLS / "gguf-py"
        if dest_gguf.exists():
            shutil.rmtree(dest_gguf)
        shutil.copytree(gguf_py, dest_gguf)
    shutil.rmtree(extract_dir, ignore_errors=True)
    if not _converter_ready(LLAMA_CPP_TOOLS):
        raise RuntimeError("Failed to install llama.cpp conversion package.")
    return LLAMA_CPP_TOOLS / "convert_hf_to_gguf.py"


def ensure_convert_script() -> Path:
    ensure_dirs()
    extra = os.environ.get("LLAMA_CPP_DIR")
    if extra and _converter_ready(Path(extra)):
        return Path(extra) / "convert_hf_to_gguf.py"
    if _converter_ready(LLAMA_CPP_TOOLS):
        return LLAMA_CPP_TOOLS / "convert_hf_to_gguf.py"
    tag = _existing_binary_tag()
    if not tag:
        tag, _url = _pick_release_asset()
    return _hydrate_converter(tag)


def ensure_quantize() -> Path:
    ensure_dirs()
    found = find_exe(["llama-quantize", "quantize"])
    if found:
        return found
    tag, url = _pick_release_asset()
    archive = LLAMA_CPP_TOOLS / Path(url).name
    print(f"Downloading llama.cpp CPU binaries from {url}", flush=True)
    _download(url, archive)
    unpacked = LLAMA_CPP_TOOLS / "bin"
    unpacked.mkdir(parents=True, exist_ok=True)
    _extract(archive, unpacked)
    found = find_exe(["llama-quantize", "quantize"], extra_roots=[unpacked])
    if not found:
        raise RuntimeError(f"Downloaded {archive.name} but llama-quantize was not inside it.")
    if not _converter_ready(LLAMA_CPP_TOOLS):
        _hydrate_converter(tag)
    return found
