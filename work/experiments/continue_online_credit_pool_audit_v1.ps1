param()

$ErrorActionPreference = 'Stop'
$poolAuditRoot = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$poolAuditPython = 'C:\Users\callofthenight\AppData\Local\Programs\Python\Python314\python.exe'
if (-not (Test-Path -LiteralPath $poolAuditPython)) { throw 'Verified Python runtime unavailable.' }
Set-Location -LiteralPath $poolAuditRoot
foreach ($poolAuditPhase in @('predictions','evaluation','audit')) {
    $poolAuditFolder = Join-Path $poolAuditRoot "results\online_credit_fresh_pilot\pilot_${poolAuditPhase}_v2"
    $poolAuditSummary = Join-Path $poolAuditFolder 'summary.json'
    if ((-not (Test-Path -LiteralPath $poolAuditSummary)) -or (Test-Path -LiteralPath (Join-Path $poolAuditFolder 'failure.json'))) {
        throw "Complete original pilot stage required before supplemental audit: $poolAuditPhase"
    }
    if (-not (Get-Content -LiteralPath $poolAuditSummary -Raw | ConvertFrom-Json).passed) {
        throw "Original pilot did not pass: $poolAuditPhase"
    }
}
foreach ($poolAuditPhase in @('preflight','pilot')) {
    if (Test-Path -LiteralPath (Join-Path $poolAuditRoot "results\online_credit_fresh_pilot\${poolAuditPhase}_pool_audit_v1")) {
        throw "Supplemental output already exists: $poolAuditPhase. Inspect and preserve it, no automatic retry."
    }
}
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
& $poolAuditPython (Join-Path $PSScriptRoot 'audit_online_credit_fresh_pools_v1.py') --selftest
if ($LASTEXITCODE -ne 0) { throw 'Supplemental self-test failed; do not continue.' }
foreach ($poolAuditPhase in @('preflight','pilot')) {
    Write-Output ([pscustomobject]@{Phase=$poolAuditPhase; Event='starting_support_pool_audit'} | ConvertTo-Json -Compress)
    & $poolAuditPython (Join-Path $PSScriptRoot 'audit_online_credit_fresh_pools_v1.py') --stage $poolAuditPhase
    if ($LASTEXITCODE -ne 0) { throw "Supplemental audit failed: $poolAuditPhase. Preserve outputs and source freeze." }
}
