# DBB Benchmark - download the 8 Ollama models used in the experiments.
# Run from any folder: powershell -ExecutionPolicy Bypass -File .\Code\download_models.ps1
param(
    [switch]$OnlyMissing
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
# If the script is downloaded outside Code/, assume its parent is the project root.
if (-not (Test-Path (Join-Path $projectRoot 'Code'))) {
    $projectRoot = $PSScriptRoot
}
$modelDir = Join-Path $projectRoot 'llm'
New-Item -ItemType Directory -Force -Path $modelDir | Out-Null

$ollamaCmd = Get-Command ollama -ErrorAction SilentlyContinue
if ($ollamaCmd) {
    $ollama = $ollamaCmd.Source
} else {
    $ollama = Join-Path $env:LOCALAPPDATA 'Programs\Ollama\ollama.exe'
    if (-not (Test-Path $ollama)) {
        Write-Host 'Khong tim thay Ollama. Cai Ollama tai https://ollama.com/download/windows' -ForegroundColor Red
        exit 1
    }
}

$env:OLLAMA_MODELS = $modelDir
Write-Host "Thu muc model: $modelDir" -ForegroundColor Cyan
Write-Host 'LUU Y: Neu Ollama da chay truoc khi thiet lap OLLAMA_MODELS, hay thoat Ollama hoan toan va khoi dong lai de server su dung dung thu muc.' -ForegroundColor Yellow

$models = @(
    'qwen3:0.6b',
    'qwen3:1.7b',
    'llama3.2:1b',
    'gemma3:1b',
    'phi3:mini',
    'smollm2:1.7b',
    'stablelm2:1.6b',
    'tinyllama:1.1b'
)

# The Ollama server must be running before pull commands.
try {
    $null = Invoke-RestMethod -Uri 'http://127.0.0.1:11434/api/tags' -TimeoutSec 5
} catch {
    Write-Host 'Ollama server chua chay. Hay mo ung dung Ollama roi chay lai script.' -ForegroundColor Red
    exit 1
}

$installed = @()
if ($OnlyMissing) {
    $installed = @(& $ollama list 2>$null | Select-Object -Skip 1 | ForEach-Object { ($_ -split '\s+')[0] })
}

$failed = @()
foreach ($model in $models) {
    if ($OnlyMissing -and ($installed -contains $model)) {
        Write-Host "Bo qua (da co): $model" -ForegroundColor DarkGray
        continue
    }
    Write-Host "`nDang tai: $model" -ForegroundColor Green
    & $ollama pull $model
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "Tai that bai: $model"
        $failed += $model
    }
}

Write-Host "`nDanh sach model hien co:" -ForegroundColor Cyan
& $ollama list
if ($failed.Count -gt 0) {
    Write-Host ('Model tai loi: ' + ($failed -join ', ')) -ForegroundColor Red
    exit 1
}
Write-Host 'Hoan tat. Ban co the chay website DBB.' -ForegroundColor Green
