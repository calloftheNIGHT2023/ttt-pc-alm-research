$ErrorActionPreference='Stop'
$reinvestmentProject=Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$reinvestmentPython='C:\Users\callofthenight\AppData\Local\Programs\Python\Python314\python.exe'
Set-Location -LiteralPath $reinvestmentProject
$env:OPENBLAS_NUM_THREADS='1'
$env:OMP_NUM_THREADS='1'
foreach ($reinvestmentScript in @('test_budget_reinvestment_v1.py','calibrate_budget_reinvestment_v1.py','audit_budget_reinvestment_v1.py')) {
    Write-Output ([pscustomobject]@{ Script=$reinvestmentScript; Event='starting' } | ConvertTo-Json -Compress)
    & $reinvestmentPython (Join-Path $PSScriptRoot $reinvestmentScript)
    if ($LASTEXITCODE -ne 0) { throw "Stage failed: $reinvestmentScript; outputs preserved, no automatic retry." }
    Write-Output ([pscustomobject]@{ Script=$reinvestmentScript; Event='completed' } | ConvertTo-Json -Compress)
}
