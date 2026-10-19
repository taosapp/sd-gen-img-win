# 一键环境准备：创建 conda 环境、安装 CUDA 版 PyTorch 与生图依赖。
#
# 用法：
#   powershell -ExecutionPolicy Bypass -File setup.ps1
#   powershell -ExecutionPolicy Bypass -File setup.ps1 -SkipModels   # 不下载权重
#
# 说明：
#   * torch 的 Windows CUDA 轮子不在 PyPI 上，必须走 download.pytorch.org 的索引。
#   * 本机已验证 torch 2.14.0+cu126 可用（另一套 comfyui 环境同版本），故对齐该版本。

param(
    [string]$EnvName = 'genimg',
    [string]$TorchVersion = '2.14.0',
    [string]$TorchVisionVersion = '0.29.0',
    [string]$CudaTag = 'cu126',
    [switch]$Recreate,
    [switch]$SkipModels,
    [switch]$SkipTorch
)

$ErrorActionPreference = 'Stop'
$ScriptRoot = if ($PSCommandPath) { Split-Path -Parent $PSCommandPath } else { $PSScriptRoot }
$ProjectRoot = $ScriptRoot
. (Join-Path $ScriptRoot 'scripts\common.ps1')

$PythonVersion = '3.12'
$CondaExe = Get-CondaExe
if (-not $CondaExe) {
    throw "未找到 conda。请先安装 Miniconda/Anaconda，或设置环境变量 GENIMG_PYTHON 指向解释器。"
}
Write-Host "[setup] conda: $CondaExe" -ForegroundColor Cyan

$existing = & $CondaExe env list | Select-String -Pattern "^\s*$EnvName\s"
if ($existing -and $Recreate) {
    Write-Host "[setup] 删除已有环境 $EnvName ..." -ForegroundColor Yellow
    & $CondaExe env remove -n $EnvName -y
    $existing = $null
}

if (-not $existing) {
    Write-Host "[setup] 创建 conda 环境 $EnvName (Python $PythonVersion) ..." -ForegroundColor Cyan
    & $CondaExe create -n $EnvName "python=$PythonVersion" -y
    if ($LASTEXITCODE -ne 0) { throw "conda create 失败" }
}
else {
    Write-Host "[setup] 环境 $EnvName 已存在，跳过创建（需要重建请加 -Recreate）" -ForegroundColor Cyan
}

$Python = Get-EnvPython -CondaExe $CondaExe -EnvName $EnvName
if (-not $Python) { throw "找不到 $EnvName 环境的 python.exe" }
Write-Host "[setup] 解释器: $Python" -ForegroundColor Cyan
$env:GENIMG_PYTHON = $Python

if (-not $SkipTorch) {
    Write-Host "[setup] 安装 CUDA 版 PyTorch ($TorchVersion+$CudaTag) ..." -ForegroundColor Cyan
    Invoke-EnvPython -Python $Python -Arguments @(
        '-m', 'pip', 'install', '--index-url', "https://download.pytorch.org/whl/$CudaTag",
        "torch==$TorchVersion", "torchvision==$TorchVisionVersion"
    )
    if ($LASTEXITCODE -ne 0) { throw "PyTorch 安装失败（退出码 $LASTEXITCODE）" }
}
else {
    Write-Host "[setup] 跳过 PyTorch 安装" -ForegroundColor Yellow
}

Write-Host "[setup] 安装生图依赖 ..." -ForegroundColor Cyan
Invoke-EnvPython -Python $Python -Arguments @('-m', 'pip', 'install', '-r', (Join-Path $ProjectRoot 'requirements.txt'))
if ($LASTEXITCODE -ne 0) { throw "依赖安装失败（退出码 $LASTEXITCODE）" }

Write-Host "[setup] 校验环境 ..." -ForegroundColor Cyan
Invoke-EnvPython -Python $Python -Arguments @('-c', "import torch, diffusers, transformers; print('torch', torch.__version__, '| cuda', torch.version.cuda, '| 可用', torch.cuda.is_available(), '| diffusers', diffusers.__version__, '| transformers', transformers.__version__)")

if (-not $SkipModels) {
    Write-Host "[setup] 下载模型权重（约 16GB，可中断后续传）..." -ForegroundColor Cyan
    Invoke-EnvPython -Python $Python -Arguments @((Join-Path $ProjectRoot 'tools\fetch_models.py')) | Out-Null
}

Write-Host ""
Write-Host "[setup] 完成。试试：" -ForegroundColor Green
Write-Host "  powershell -File run.ps1 --list" -ForegroundColor Green
Write-Host "  powershell -File run.ps1 `"a red panda astronaut, cinematic`" --size 512x512" -ForegroundColor Green
