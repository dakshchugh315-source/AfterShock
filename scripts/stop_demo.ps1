# AfterShock: stop the public link (and, with -All, the dashboard too).
param([switch]$All)
Get-Process cloudflared -ErrorAction SilentlyContinue | Stop-Process -Force
Write-Host "Public link stopped."
if ($All) {
    Get-Process streamlit -ErrorAction SilentlyContinue | Stop-Process -Force
    Write-Host "Dashboard stopped."
}
