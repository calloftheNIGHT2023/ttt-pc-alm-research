$ErrorActionPreference='Stop'
$radiusOnlineProject=Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$radiusOnlinePython='C:\Users\callofthenight\AppData\Local\Programs\Python\Python314\python.exe'
Set-Location -LiteralPath $radiusOnlineProject
$env:OPENBLAS_NUM_THREADS='1'
$env:OMP_NUM_THREADS='1'
$radiusOnlineStages=@(
    [pscustomobject]@{ File='run_search_radius_development_v1.py'; Extra=@('--stage','preflight') },
    [pscustomobject]@{ File='run_search_radius_development_v1.py'; Extra=@('--stage','development') },
    [pscustomobject]@{ File='evaluate_search_radius_development_v1.py'; Extra=@() },
    [pscustomobject]@{ File='audit_search_radius_development_v1.py'; Extra=@() }
)
foreach ($radiusOnlineStage in $radiusOnlineStages) {
    $radiusOnlineArgs=$radiusOnlineStage.Extra
    Write-Output ([pscustomobject]@{ Script=$radiusOnlineStage.File; Arguments=$radiusOnlineArgs; Event='starting' } | ConvertTo-Json -Compress)
    & $radiusOnlinePython (Join-Path $PSScriptRoot $radiusOnlineStage.File) @radiusOnlineArgs
    if ($LASTEXITCODE -ne 0) { throw "Stage failed: $($radiusOnlineStage.File); outputs preserved, no automatic retry." }
    Write-Output ([pscustomobject]@{ Script=$radiusOnlineStage.File; Event='completed' } | ConvertTo-Json -Compress)
}
