param(
    [string]$Python = (Join-Path $PSScriptRoot '.venv\Scripts\python.exe'),
    [string]$NYT = (Join-Path $PSScriptRoot 'HW-1\nyt.csv'),
    [string]$AG = (Join-Path $PSScriptRoot 'HW-1\ag.csv')
)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
if (-not (Test-Path -LiteralPath $Python)) { $Python = 'python' }
$env:PYTHONHASHSEED = '42'
$env:PYTHONIOENCODING = 'utf-8'
$env:OMP_NUM_THREADS = '4'
$env:MKL_NUM_THREADS = '4'
$env:TOKENIZERS_PARALLELISM = 'false'

function Invoke-HomeworkPython([string[]]$Arguments) {
    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Python failed with exit code $LASTEXITCODE" }
}

Invoke-HomeworkPython -Arguments @('download_resources.py', 'all')
Invoke-HomeworkPython -Arguments @('experiment.py', '--task', 'all', '--nyt', $NYT, '--ag', $AG)
Invoke-HomeworkPython -Arguments @('build_report.py', '--nyt', $NYT)
Invoke-HomeworkPython -Arguments @('-m', 'pytest', 'tests', '-q')
Write-Host 'Finished. See 实验报告.pdf and outputs/results.csv.'
