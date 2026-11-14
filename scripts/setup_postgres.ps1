# One-time PostgreSQL setup for AfterShock (run in a normal PowerShell window, from C:\Users\DAKSH\AfterShock):
#   powershell -ExecutionPolicy Bypass -File scripts\setup_postgres.ps1
# psql will ask for YOUR postgres superuser password (the one chosen when PostgreSQL was installed).
# It creates a separate login role "aftershock" (its own random password, kept in .env, which git ignores),
# a database "aftershock" owned by that role, and the extensions the schema uses.
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$psql = "C:\Program Files\PostgreSQL\18\bin\psql.exe"
if (-not (Test-Path $psql)) { Write-Host "psql not found at $psql" -ForegroundColor Red; exit 1 }

$envFile = Join-Path $root ".env"
if (-not (Test-Path $envFile) -or -not (Select-String -Path $envFile -Pattern "^AFTERSHOCK_PG=" -Quiet)) {
    # random password for the app role (letters + digits only, so it is safe inside a URL)
    $chars = [char[]]"abcdefghijkmnopqrstuvwxyzABCDEFGHJKLMNPQRSTUVWXYZ23456789"
    $pw = -join (1..24 | ForEach-Object { $chars | Get-Random })
    Add-Content -Path $envFile -Value "AFTERSHOCK_PG=postgresql://aftershock:$pw@localhost:5432/aftershock" -Encoding ascii
}
$pg = (Select-String -Path $envFile -Pattern "^AFTERSHOCK_PG=(.*)$").Matches[0].Groups[1].Value
$pw = ([uri]$pg).UserInfo.Split(":")[1]

$sql = @"
DO `$`$ BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'aftershock') THEN
    CREATE ROLE aftershock LOGIN PASSWORD '$pw';
  ELSE
    ALTER ROLE aftershock LOGIN PASSWORD '$pw';
  END IF;
END `$`$;
SELECT 'CREATE DATABASE aftershock OWNER aftershock'
WHERE NOT EXISTS (SELECT 1 FROM pg_database WHERE datname = 'aftershock')\gexec
\c aftershock
CREATE EXTENSION IF NOT EXISTS cube;
CREATE EXTENSION IF NOT EXISTS earthdistance;
DO `$`$ BEGIN
  IF EXISTS (SELECT 1 FROM pg_available_extensions WHERE name = 'postgis') THEN
    CREATE EXTENSION IF NOT EXISTS postgis;
    RAISE NOTICE 'PostGIS enabled';
  ELSE
    RAISE NOTICE 'PostGIS not installed: using earthdistance for spatial queries (install PostGIS via Stack Builder to switch)';
  END IF;
END `$`$;
GRANT ALL ON SCHEMA public TO aftershock;
"@
$tmp = Join-Path $env:TEMP "aftershock_setup.sql"
Set-Content -Path $tmp -Value $sql -Encoding ascii
& $psql -h localhost -p 5432 -U postgres -d postgres -v ON_ERROR_STOP=1 -f $tmp
$code = $LASTEXITCODE
Remove-Item $tmp -Force
if ($code -ne 0) { Write-Host "Setup failed (see the message above)." -ForegroundColor Red; exit $code }
Write-Host "Done. Role and database 'aftershock' are ready. Next: python scripts/load_postgres.py" -ForegroundColor Green
