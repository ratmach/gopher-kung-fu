$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

if (-not (Test-Path ".venv")) {
  python -m venv .venv
}
& .\.venv\Scripts\python -m pip install -e .

Start-Process -NoNewWindow -FilePath ".\.venv\Scripts\python.exe" -ArgumentList "-m", "uvicorn", "app.main:app", "--reload", "--host", "127.0.0.1", "--port", "8000", "--reload-exclude", "data", "--reload-exclude", "unsloth_compiled_cache", "--reload-exclude", ".venv", "--reload-exclude", "web"
Set-Location "$root\web"
if (-not (Test-Path "node_modules")) {
  npm install
}
npm run dev
