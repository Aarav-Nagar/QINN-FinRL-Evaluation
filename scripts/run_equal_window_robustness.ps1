param(
    [string]$OutputRoot
)

$ErrorActionPreference = "Stop"
$Repository = Split-Path -Parent $PSScriptRoot
$ResearchRoot = Split-Path -Parent (Split-Path -Parent $Repository)
$WorkRoot = Join-Path $ResearchRoot "work"
if (-not $OutputRoot) {
    $OutputRoot = Join-Path $WorkRoot "equal_window_robustness_2026-07-31"
}

New-Item -ItemType Directory -Path $OutputRoot -Force | Out-Null
$env:PYTHONPATH = Join-Path $WorkRoot "pydeps"

Push-Location $Repository
try {
    foreach ($Window in @("equal_2019_2020", "equal_2021_2022")) {
        $WindowRoot = Join-Path $OutputRoot $Window
        $LogPath = Join-Path $OutputRoot "$Window.log"
        New-Item -ItemType Directory -Path $WindowRoot -Force | Out-Null
        & py -3.12 scripts\run_experiment_matrix.py `
            --phase equal-window `
            --window $Window `
            --data-dir (Join-Path $WorkRoot "data") `
            --finrl-dir (Join-Path $WorkRoot "FinRL") `
            --output-root $WindowRoot `
            --timesteps 20000 `
            --bond-dimensions 2 `
            --seeds 0 1 2 3 4 5 6 7 8 9 `
            --encoder-epochs 60 `
            --encoder-patience 10 `
            --encoder-batch-size 512 `
            --encoder-device cpu *>> $LogPath
        if ($LASTEXITCODE -ne 0) {
            throw "$Window matrix failed with exit code $LASTEXITCODE"
        }
    }
}
finally {
    Pop-Location
}
