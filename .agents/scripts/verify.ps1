param(
    [string]$Python = $env:AGENT_PYTHON
)
$ErrorActionPreference = 'Stop'

if ([string]::IsNullOrWhiteSpace($Python)) {
    $Python = (Get-Command python -ErrorAction Stop).Source
}

function Invoke-CheckedPython {
    param([string[]]$Arguments)
    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Python 命令失败（exit=$LASTEXITCODE）: $Arguments"
    }
}

Invoke-CheckedPython -Arguments @('.agents/scripts/data_manifest.py', 'validate')
Invoke-CheckedPython -Arguments @('.agents/scripts/validate_loop.py')
Invoke-CheckedPython -Arguments @('acceptance-pipeline/parse_specs.py')
Invoke-CheckedPython -Arguments @('acceptance-pipeline/generate_tests.py')
Invoke-CheckedPython -Arguments @('-m', 'pytest', 'tests', 'generated-acceptance-tests', '-q')
Invoke-CheckedPython -Arguments @('.agents/agentctl.py', 'doctor')
