$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

if (-not (Test-Path ".\.venv\Scripts\python.exe")) {
  throw "Create .venv first (python -m venv .venv)."
}

$py = ".\.venv\Scripts\python.exe"
$index = "https://download.pytorch.org/whl/cu130"

Write-Host "Installing CUDA PyTorch from $index …"
& $py -m pip install --upgrade torch torchvision --index-url $index
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "Installing Unsloth + datasets …"
& $py -m pip install --upgrade unsloth datasets
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

# `pip install unsloth` can replace CUDA torch with the CPU wheel from PyPI.
Write-Host "Reasserting CUDA PyTorch …"
& $py -m pip install --upgrade torch torchvision --index-url $index
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $py -c "import torch; assert torch.cuda.is_available(), f'CUDA still unavailable: {torch.__version__} cuda={torch.version.cuda}'; print(torch.cuda.get_device_name(0), torch.__version__, 'cuda', torch.version.cuda)"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
