param([string]$Python = "")
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
if (-not $Python) {
    foreach ($name in @('python', 'python3', 'python3.12', 'py')) {
        $command = Get-Command $name -ErrorAction SilentlyContinue
        if ($command) { $Python = $command.Source; break }
    }
}
if (-not $Python) {
    $managed = Join-Path $env:APPDATA 'uv\python'
    if (Test-Path -LiteralPath $managed) {
        $installed = Get-ChildItem -LiteralPath $managed -Directory |
            Where-Object { $_.Name -like 'cpython-3.1*-windows-x86_64-none' } |
            Sort-Object Name -Descending | Select-Object -First 1
        if ($installed) { $Python = Join-Path $installed.FullName 'python.exe' }
    }
}
if (-not $Python) { throw 'Python 3.11+ is required. Pass -Python C:\path\to\python.exe.' }
& $Python -c "import sys; assert sys.version_info >= (3,11), 'Python 3.11+ required'"
if ($LASTEXITCODE -ne 0) { throw 'Python version check failed.' }
# Standard-library project: pip is unnecessary. This also avoids ensurepip restrictions.
& $Python -m venv --without-pip .venv
if ($LASTEXITCODE -ne 0) { throw 'Virtual environment creation failed.' }
& '.\.venv\Scripts\python.exe' run_demo.py
if ($LASTEXITCODE -ne 0) { throw 'Demo failed; inspect submission_artifacts\run.log.' }
