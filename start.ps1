param([switch]$NoBrowser)

# Starts the API (port 8000) and the web app (port 5173), each in its own window. Close the windows to stop.
$root = $PSScriptRoot
Set-Location $root

if (-not (Test-Path "$root\.venv\Scripts\python.exe")) {
    Write-Host "Missing .venv. Create it first (see README Quick start)." -ForegroundColor Red
    exit 1
}
if (-not (Test-Path "$root\web\node_modules")) {
    Write-Host "Installing web dependencies (first run only)..."
    Push-Location "$root\web"; npm install; Pop-Location
}

Start-Process powershell -ArgumentList "-NoExit", "-Command", "`$host.UI.RawUI.WindowTitle='API :8000'; Set-Location '$root'; `$env:PYTHONPATH='src'; .\.venv\Scripts\python -m uvicorn api.main:app --port 8000"
Start-Process powershell -ArgumentList "-NoExit", "-Command", "`$host.UI.RawUI.WindowTitle='WEB :5173'; Set-Location '$root\web'; npm run dev"

Write-Host "Waiting for the API..."
$ready = $false
for ($i = 0; $i -lt 60 -and -not $ready; $i++) {
    Start-Sleep -Seconds 1
    try { $ready = (Invoke-WebRequest -Uri "http://127.0.0.1:8000/health" -UseBasicParsing -TimeoutSec 2).StatusCode -eq 200 } catch {}
}
if (-not $ready) { Write-Host "API did not start in 60 s. Check the API window for errors." -ForegroundColor Red; exit 1 }

Write-Host "API ready:  http://127.0.0.1:8000/docs"
Write-Host "Web app:    http://localhost:5173"
if (-not $NoBrowser) { Start-Sleep -Seconds 2; Start-Process "http://localhost:5173" }
