$ErrorActionPreference = "Stop"
$Root = $PSScriptRoot
$ModelDir = Join-Path $Root "llm"
New-Item -ItemType Directory -Force -Path $ModelDir | Out-Null
[Environment]::SetEnvironmentVariable("OLLAMA_MODELS", $ModelDir, "User")
$env:OLLAMA_MODELS = $ModelDir
Write-Host "OLLAMA_MODELS = $ModelDir"
Write-Host "Exit Ollama completely (tray icon), then restart Ollama before pulling models."
Write-Host "Models are NOT included in this ZIP."
