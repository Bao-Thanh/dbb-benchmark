$ErrorActionPreference = "Stop"
$env:OLLAMA_MODELS = Join-Path $PSScriptRoot "llm"
$ollama = (Get-Command ollama -ErrorAction SilentlyContinue).Source
if (-not $ollama) { $ollama = Join-Path $env:LOCALAPPDATA "Programs\Ollama\ollama.exe" }
if (-not (Test-Path $ollama)) { throw "Please install Ollama first: https://ollama.com/download/windows" }
$models = @("qwen3:0.6b", "qwen3:1.7b", "llama3.2:1b", "gemma3:1b", "phi3:mini", "smollm2:1.7b", "stablelm2:1.6b", "tinyllama:1.1b")
foreach ($model in $models) { & $ollama pull $model; if ($LASTEXITCODE -ne 0) { throw "Failed: $model" } }
& $ollama list
