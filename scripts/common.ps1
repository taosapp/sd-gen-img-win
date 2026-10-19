# 共用工具：定位项目根目录、conda、目标环境与解释器。
# 由 setup.ps1 与 run.ps1 点源引入，不单独执行。
#
# 注意：不要依赖 $PSScriptRoot —— 被点源的脚本里它可能是空值
# （实测在 `powershell -File run.ps1` 场景下为空，导致 Split-Path 绑定失败）。
# 统一用 $PSCommandPath 推导路径。

$ErrorActionPreference = 'Stop'

# 调用方通常已设置 $ProjectRoot；没有则自己推导。
if (-not $ProjectRoot) {
    if ($PSCommandPath) {
        # 本文件位于 <root>\scripts\common.ps1
        $ProjectRoot = Split-Path -Parent (Split-Path -Parent $PSCommandPath)
    }
    elseif ($PSScriptRoot) {
        $ProjectRoot = Split-Path -Parent $PSScriptRoot
    }
    else {
        $ProjectRoot = (Get-Location).Path
    }
}
$script:ProjectRoot = $ProjectRoot

if (-not $EnvName) {
    $EnvName = if ($env:GENIMG_ENV) { $env:GENIMG_ENV } else { 'genimg' }
}
$script:EnvName = $EnvName

function Get-CondaExe {
    # 1) PATH 里能找到就用
    $cmd = Get-Command conda -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }

    # 2) 常见安装位置。注意：环境变量可能缺失（实测本机 $env:LOCALAPPDATA 为空），
    #    所以每个候选都必须先判空，不能直接 Join-Path。
    $bases = @($env:USERPROFILE, $env:LOCALAPPDATA, $env:ProgramData) | Where-Object { $_ }
    foreach ($b in $bases) {
        foreach ($sub in @('miniconda3', 'anaconda3')) {
            $p = Join-Path $b "$sub\Scripts\conda.exe"
            if (Test-Path $p) { return $p }
        }
    }
    foreach ($p in @(
            'C:\ProgramData\miniconda3\Scripts\conda.exe',
            'C:\ProgramData\Anaconda3\Scripts\conda.exe',
            'D:\miniconda3\Scripts\conda.exe',
            'D:\Anaconda3\Scripts\conda.exe',
            'C:\ProgramData\Miniconda3\Scripts\conda.exe'
        )) {
        if (Test-Path $p) { return $p }
    }
    return $null
}

function Get-EnvPython {
    param([string]$CondaExe, [string]$EnvName)

    # 1) 显式指定优先
    if ($env:GENIMG_PYTHON -and (Test-Path $env:GENIMG_PYTHON)) {
        return $env:GENIMG_PYTHON
    }

    # 2) conda 环境的常规布局（base 的 envs 子目录 / 用户目录）
    $roots = @()
    if ($CondaExe) { $roots += (Split-Path -Parent (Split-Path -Parent $CondaExe)) }
    foreach ($b in @($env:USERPROFILE, $env:LOCALAPPDATA)) {
        if (-not $b) { continue }
        $roots += (Join-Path $b 'miniconda3')
        $roots += (Join-Path $b 'anaconda3')
    }
    foreach ($r in $roots) {
        if (-not $r) { continue }
        $p = Join-Path $r "envs\$EnvName\python.exe"
        if (Test-Path $p) { return $p }
    }

    # 3) 项目内 venv 兜底
    if ($ProjectRoot) {
        $v = Join-Path $ProjectRoot '.venv\Scripts\python.exe'
        if (Test-Path $v) { return $v }
    }

    return $null
}

function Invoke-EnvPython {
    param([string]$Python, [string[]]$Arguments)

    # 关键：AutoClaw 会注入 PIP_USER=1 与 PYTHONUSERBASE，导致 pip 把包装到用户目录
    # 而不是当前环境（实测会污染 C:\Users\<用户>\AppData\Roaming\AutoClaw-official\...）。
    # 这里显式压掉，保证依赖装在目标环境内、也只从目标环境读取。
    $oldPipUser = $env:PIP_USER
    $oldUserBase = $env:PYTHONUSERBASE
    $oldNoSite = $env:PYTHONNOUSERSITE
    $env:PIP_USER = '0'
    $env:PYTHONNOUSERSITE = '1'
    Remove-Item Env:PYTHONUSERBASE -ErrorAction SilentlyContinue
    try {
        # 不 return 退出码：函数一旦 return，调用方用 $code = 赋值时会把 Python 的 stdout
        # 一并吞掉。调用方改为读 $LASTEXITCODE（原生命令执行后由 PowerShell 自动维护）。
        & $Python @Arguments
    }
    finally {
        if ($null -eq $oldPipUser) { Remove-Item Env:PIP_USER -ErrorAction SilentlyContinue } else { $env:PIP_USER = $oldPipUser }
        if ($null -eq $oldUserBase) { Remove-Item Env:PYTHONUSERBASE -ErrorAction SilentlyContinue } else { $env:PYTHONUSERBASE = $oldUserBase }
        if ($null -eq $oldNoSite) { Remove-Item Env:PYTHONNOUSERSITE -ErrorAction SilentlyContinue } else { $env:PYTHONNOUSERSITE = $oldNoSite }
    }
}
