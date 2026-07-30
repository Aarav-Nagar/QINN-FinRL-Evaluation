param(
    [string]$OutputRoot
)

$ErrorActionPreference = "Stop"
$Repository = Split-Path -Parent $PSScriptRoot
$ResearchRoot = Split-Path -Parent (Split-Path -Parent $Repository)
$WorkRoot = Join-Path $ResearchRoot "work"
if (-not $OutputRoot) {
    $OutputRoot = Join-Path $WorkRoot "temporal_robustness_2026-07-30"
}

New-Item -ItemType Directory -Path $OutputRoot -Force | Out-Null
$env:PYTHONPATH = Join-Path $WorkRoot "pydeps"
$LogPath = Join-Path $OutputRoot "matrix.log"

Push-Location $Repository
try {
    & py -3.12 scripts\run_experiment_matrix.py `
        --phase temporal-robustness `
        --window shifted `
        --data-dir (Join-Path $WorkRoot "data") `
        --finrl-dir (Join-Path $WorkRoot "FinRL") `
        --output-root $OutputRoot `
        --timesteps 20000 `
        --bond-dimensions 2 `
        --seeds 0 1 2 3 4 5 6 7 8 9 `
        --encoder-epochs 60 `
        --encoder-patience 10 `
        --encoder-batch-size 512 `
        --encoder-device cpu *>> $LogPath
    if ($LASTEXITCODE -ne 0) {
        throw "Temporal robustness matrix failed with exit code $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}
