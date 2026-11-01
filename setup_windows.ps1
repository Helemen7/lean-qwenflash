# One-shot Windows setup + bring-up check. Run in "x64 Native Tools Command Prompt for VS" (C++ workload installed):
#   setup_windows.bat                       auto: NVIDIA driver -> CUDA 12.8 torch, otherwise AMD Radeon RDNA3 ROCm torch
#   setup_windows.bat -Torch "<pip args>"   explicit torch install arguments
param([string]$Torch = "")
Set-Location $PSScriptRoot
$nvidia = [bool](Get-Command nvidia-smi -ErrorAction SilentlyContinue)
if (-not $Torch) {
  if ($nvidia) { $Torch = "torch --index-url https://download.pytorch.org/whl/cu128" }
  else { $Torch = "--pre --index-url https://rocm.nightlies.amd.com/v2/gfx110X-all/ torch ""rocm[libraries,devel]""" }
}
Write-Host "GPU path: $(if ($nvidia) {'NVIDIA'} else {'AMD (ROCm)'}) | torch args: $Torch"
if (-not (Get-Command cl -ErrorAction SilentlyContinue)) { Write-Warning "cl.exe not found: use an 'x64 Native Tools Command Prompt' of Visual Studio (C++ workload)." }
if (-not (Test-Path .venv)) { py -3.12 -m venv .venv; if ($LASTEXITCODE) { py -3.11 -m venv .venv } }
& .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip uninstall -y torch torchvision torchaudio 2>&1 | Out-Null
Invoke-Expression "pip install $Torch"
pip install fastapi uvicorn safetensors jinja2 huggingface_hub numpy
if ($nvidia) { pip install triton-windows }   # ROCm torch wheels bring their own Triton
pip install -e exllamav3
python windows_check.py
Write-Host "`nDone. Report: $PSScriptRoot\windows_report.txt"
