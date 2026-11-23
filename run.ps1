# Start the analyzer (default http://127.0.0.1:8091; override with PITCHLAB_HOST / PITCHLAB_PORT in .env).
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
$env:PYTHONUTF8 = '1'
& .\.venv\Scripts\python.exe -m pitchlab.server
