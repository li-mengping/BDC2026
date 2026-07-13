$ErrorActionPreference = 'Stop'
function Invoke-ProjectPython {
    param([string[]]$CommandArgs)
    if ($env:BDC_PYTHON) {
        & $env:BDC_PYTHON @CommandArgs
    } elseif (Get-Command py -ErrorAction SilentlyContinue) {
        & py -3.12 @CommandArgs
    } else {
        & python @CommandArgs
    }
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
}

Invoke-ProjectPython -CommandArgs @('acceptance-pipeline/parse_specs.py')
Invoke-ProjectPython -CommandArgs @('acceptance-pipeline/generate_tests.py')
Invoke-ProjectPython -CommandArgs @('-m', 'pytest', 'generated-acceptance-tests', '-q')
