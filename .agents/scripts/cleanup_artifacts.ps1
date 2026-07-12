$ErrorActionPreference = 'Stop'
$root = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
$metadataPath = Join-Path $root 'model/xgboost/metadata.json'
$modelRootPath = Join-Path $root 'model/xgboost/models'
if (-not (Test-Path -LiteralPath $metadataPath) -or -not (Test-Path -LiteralPath $modelRootPath)) {
    throw '缺少 metadata 或模型目录，拒绝猜测清理范围'
}
$modelRoot = (Resolve-Path -LiteralPath $modelRootPath).Path
$metadata = Get-Content -Raw -LiteralPath $metadataPath | ConvertFrom-Json
$keep = [System.Collections.Generic.HashSet[string]]::new([System.StringComparer]::OrdinalIgnoreCase)
foreach ($model in $metadata.models) {
    $artifact = [System.IO.Path]::GetFullPath((Join-Path (Split-Path $metadataPath) $model.path))
    if (-not $artifact.StartsWith($modelRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "metadata 引用了模型目录外文件: $artifact"
    }
    [void]$keep.Add($artifact)
    $artifactDirectory = [System.IO.Path]::GetDirectoryName($artifact)
    $artifactStem = [System.IO.Path]::GetFileNameWithoutExtension($artifact)
    [void]$keep.Add([System.IO.Path]::Combine($artifactDirectory, $artifactStem + '_evals.json'))
}
Get-ChildItem -LiteralPath $modelRoot -File | ForEach-Object {
    if (-not $_.FullName.StartsWith($modelRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "拒绝删除工作区外文件: $($_.FullName)"
    }
    if (-not $keep.Contains($_.FullName)) {
        Remove-Item -LiteralPath $_.FullName -Force
    }
}
