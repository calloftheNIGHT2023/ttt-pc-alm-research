param(
    [Parameter(Mandatory=$true)]
    [ValidateSet('preflight','pilot')]
    [string]$Phase
)

$ErrorActionPreference='Stop'
$experimentProject=Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$experimentPython='C:\Users\callofthenight\AppData\Local\Programs\Python\Python314\python.exe'
if (-not (Test-Path -LiteralPath $experimentPython)) { throw 'Verified Python runtime is unavailable.' }
Set-Location -LiteralPath $experimentProject
$env:OPENBLAS_NUM_THREADS='1'
$env:OMP_NUM_THREADS='1'

# Source gates, exclusive output directories and each phase's summary prevent
# silent retries, partial-query scoring or a skipped preflight audit.
foreach ($experimentScript in @('run_online_credit_fresh_v2.py','evaluate_online_credit_fresh_v2.py','audit_online_credit_fresh_v2.py')) {
    Write-Output ([pscustomobject]@{ Phase=$Phase; Script=$experimentScript; Event='starting' } | ConvertTo-Json -Compress)
    & $experimentPython (Join-Path $PSScriptRoot $experimentScript) --stage $Phase
    if ($LASTEXITCODE -ne 0) {
        throw "Stage failed: $experimentScript, exit code $LASTEXITCODE. Preserve its existing outputs; no automatic retry."
    }
    Write-Output ([pscustomobject]@{ Phase=$Phase; Script=$experimentScript; Event='completed' } | ConvertTo-Json -Compress)
}
