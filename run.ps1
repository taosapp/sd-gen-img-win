# 用 genimg 环境的解释器运行生图 agent（自动处理 AutoClaw 注入的环境变量）。
#
# 用法：
#   powershell -File run.ps1 --list
#   powershell -File run.ps1 "a red panda astronaut" -s 512x512
#   powershell -File run.ps1 "水墨山水 [精度=fp32] [画质=max]"
#   powershell -File run.ps1 --image .\in.png --strength 0.6 "oil painting style"

[CmdletBinding()]
param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$GenImgArgs
)

$ErrorActionPreference = 'Stop'

$ScriptRoot = if ($PSCommandPath) { Split-Path -Parent $PSCommandPath } else { $PSScriptRoot }
$ProjectRoot = $ScriptRoot

. (Join-Path $ScriptRoot 'scripts\common.ps1')

$CondaExe = Get-CondaExe
$Python = Get-EnvPython -CondaExe $CondaExe -EnvName $EnvName
if (-not $Python) {
    Write-Host "找不到环境 $EnvName，请先执行： powershell -ExecutionPolicy Bypass -File setup.ps1" -ForegroundColor Red
    exit 1
}

$all = @((Join-Path $ProjectRoot 'genimg.py')) + $GenImgArgs
Invoke-EnvPython -Python $Python -Arguments $all
exit $LASTEXITCODE
