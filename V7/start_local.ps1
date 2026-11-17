param(
    [int]$Port = 8000,
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
# Run in the foreground in your signed-in Windows desktop session.
# No reload and exactly one worker: the backend owns one runner process.
& $Python -m uvicorn app:app --app-dir $PSScriptRoot --host 127.0.0.1 --port $Port --workers 1
exit $LASTEXITCODE
