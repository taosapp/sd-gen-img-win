# 启动本地 Web 界面（浏览器里上传图片 / 文生图 / 图生图）。
#
# 用法：
#   powershell -File web.ps1             # 开始服务 + 预热默认挡位（地址 http://127.0.0.1:7860）
#   powershell -File web.ps1 -Open       # 同上，并自动打开浏览器
#   powershell -File web.ps1 -Port 8080  # 换端口
#   powershell -File web.ps1 -NoPreload  # 不预热（第一次生成时才加载权重）
#   powershell -File web.ps1 -NoOpen     # 不自动开浏览器（优先级高于 -Open）
#   powershell -File web.ps1 -Help       # 查看服务端参数说明
#
# 说明：
#   * 故意不用 -Open:$false 这种写法：经 cmd/bat 转发时 PowerShell 5.1 会把它当字符串，
#     报 "cannot convert String to SwitchParameter"。所以用独立的 -NoOpen / -NoPreload。
#   * 其余未识别的参数会原样透传给 Python。
#   * 日常使用直接双击 web.bat 即可。

[CmdletBinding()]
param(
    [int]$Port = 7860,
    [string]$HostAddr = '127.0.0.1',
    [switch]$Open,
    [switch]$NoOpen,
    [switch]$NoPreload,
    [switch]$Help,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Rest
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

if ($Help) {
    $pythonArgs = @('-m', 'webapp.server', '--help')
}
else {
    $pythonArgs = @('-m', 'webapp.server', '--host', $HostAddr, '--port', "$Port")
    if ($Open -and -not $NoOpen) { $pythonArgs += '--open' }
    if (-not $NoPreload) { $pythonArgs += '--preload' }
    if ($Rest) { $pythonArgs += $Rest }
}

# -m 方式需要以项目根目录为工作目录，才能找到 webapp 包
Push-Location $ProjectRoot
try {
    Invoke-EnvPython -Python $Python -Arguments $pythonArgs
    $code = $LASTEXITCODE
}
finally {
    Pop-Location
}
exit $code
