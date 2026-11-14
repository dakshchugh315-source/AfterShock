# AfterShock: start the dashboard (+ optional public link) for a demo.
# Run from anywhere:  powershell -ExecutionPolicy Bypass -File C:\AfterShock\scripts\start_demo.ps1
# Add -Public to also open a temporary public trycloudflare.com link (anyone with the link can view it).
param([switch]$Public)

$root = Split-Path $PSScriptRoot -Parent
Set-Location $root
# The default DB in config.py is the old 3-disaster data/aftershock.db - the demo must use the full 10-disaster DB.
$env:AFTERSHOCK_DB = "sqlite:///$((Resolve-Path data).Path -replace '\\','/')/aftershock_full.db"
$streamlit = Join-Path $root ".venv\Scripts\streamlit.exe"

function Healthy($url) { try { (Invoke-WebRequest "$url/_stcore/health" -UseBasicParsing -TimeoutSec 5).Content -eq "ok" } catch { $false } }

if (Healthy "http://localhost:8501") {
    Write-Host "Dashboard already running."
} else {
    Write-Host "Starting dashboard..."
    Start-Process $streamlit -ArgumentList "run","app\dashboard.py","--server.headless","true","--server.port","8501" -WindowStyle Hidden
    foreach ($i in 1..30) { Start-Sleep 1; if (Healthy "http://localhost:8501") { break } }
}
if (-not (Healthy "http://localhost:8501")) { Write-Host "Dashboard did not start - run: $streamlit run app\dashboard.py" -ForegroundColor Red; exit 1 }
Write-Host "Local:   http://localhost:8501" -ForegroundColor Green

if ($Public) {
    $cf = Join-Path $root "runs\cloudflared.exe"
    if (-not (Test-Path $cf)) { Write-Host "cloudflared.exe not found at $cf" -ForegroundColor Red; exit 1 }
    Get-Process cloudflared -ErrorAction SilentlyContinue | Stop-Process -Force
    $errLog = Join-Path $root "runs\cf_err.log"
    Remove-Item $errLog -ErrorAction SilentlyContinue
    Start-Process $cf -ArgumentList "tunnel","--url","http://localhost:8501" -WindowStyle Hidden -RedirectStandardError $errLog -RedirectStandardOutput (Join-Path $root "runs\cf_out.log")
    $url = $null
    foreach ($i in 1..30) {
        Start-Sleep 1
        $m = Select-String -Path $errLog -Pattern "https://[a-z0-9-]+\.trycloudflare\.com" -ErrorAction SilentlyContinue
        if ($m) { $url = $m.Matches[0].Value; break }
    }
    if ($url) {
        # New quick-tunnel hostnames take ~30-60 s to resolve in DNS.
        Write-Host "Waiting for the public link to come online..."
        $ok = $false
        foreach ($i in 1..30) { Start-Sleep 3; if (Healthy $url) { $ok = $true; break } }
        if ($ok) { Write-Host "Public:  $url" -ForegroundColor Green }
        else { Write-Host "Public:  $url  (not reachable yet - give it another minute)" -ForegroundColor Yellow }
        Write-Host "(public link - stop it after the demo with scripts\stop_demo.ps1)" -ForegroundColor Yellow
    } else { Write-Host "Tunnel did not come up - check $errLog" -ForegroundColor Red }
}
