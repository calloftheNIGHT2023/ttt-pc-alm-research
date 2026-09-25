param()

$ErrorActionPreference = 'Stop'
$reportingRoot = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$reportingPython = 'C:\Users\callofthenight\AppData\Local\Programs\Python\Python314\python.exe'
if (-not (Test-Path -LiteralPath $reportingPython)) { throw 'Verified Python runtime is unavailable.' }
Set-Location -LiteralPath $reportingRoot

# This is a one-shot post-run command, not a second prediction job or monitor.
# Leave the original 328 conductor unchanged; start this only after it exits 0.
foreach ($reportingStage in @('predictions', 'evaluation', 'audit')) {
    $reportingFolder = Join-Path $reportingRoot "results\online_credit_fresh_pilot\pilot_${reportingStage}_v2"
    $reportingSummaryPath = Join-Path $reportingFolder 'summary.json'
    if ((-not (Test-Path -LiteralPath $reportingSummaryPath)) -or
        (Test-Path -LiteralPath (Join-Path $reportingFolder 'failure.json'))) {
        throw "Complete successful pilot stage required before reporting: $reportingStage"
    }
    $reportingSummary = Get-Content -LiteralPath $reportingSummaryPath -Raw | ConvertFrom-Json
    if (-not $reportingSummary.passed) { throw "Pilot stage did not pass: $reportingStage" }
}
foreach ($reportingExpected in @('preflight_report_v1', 'pilot_report_v1')) {
    if (Test-Path -LiteralPath (Join-Path $reportingRoot "results\online_credit_fresh_pilot\$reportingExpected")) {
        throw "Report output already exists. Inspect and preserve it; do not overwrite or rerun blindly: $reportingExpected"
    }
}
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
$env:MPLBACKEND = 'Agg'
& $reportingPython (Join-Path $PSScriptRoot 'test_online_credit_report_v1.py')
if ($LASTEXITCODE -ne 0) { throw 'Reporting self-test failed; no report generation attempted.' }
foreach ($reportingPhase in @('preflight', 'pilot')) {
    foreach ($reportingScript in @('report_online_credit_fresh_v1.py', 'audit_online_credit_report_v1.py')) {
        Write-Output ([pscustomobject]@{Phase=$reportingPhase; Script=$reportingScript; Event='starting'} | ConvertTo-Json -Compress)
        & $reportingPython (Join-Path $PSScriptRoot $reportingScript) --stage $reportingPhase
        if ($LASTEXITCODE -ne 0) { throw "Reporting failed: $reportingScript / $reportingPhase. Preserve outputs; do not overwrite." }
    }
}
Write-Output 'Numerical reporting is complete. Both pilot PNG files still require actual visual inspection; no automatic visual pass or research-goal completion.'
