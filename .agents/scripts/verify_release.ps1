param(
    [string]$Python = $env:AGENT_PYTHON
)
$ErrorActionPreference = 'Stop'

if ([string]::IsNullOrWhiteSpace($Python)) {
    $Python = (Get-Command python -ErrorAction Stop).Source
}
$PowerShell = (Get-Process -Id $PID).Path

& $PowerShell -NoProfile -ExecutionPolicy Bypass -File .agents/scripts/verify.ps1 -Python $Python
if ($LASTEXITCODE -ne 0) {
    throw "基础验证失败（exit=$LASTEXITCODE）"
}

& $Python .agents/scripts/release_verify.py
if ($LASTEXITCODE -ne 0) {
    throw "离线发布验证失败（exit=$LASTEXITCODE）"
}
