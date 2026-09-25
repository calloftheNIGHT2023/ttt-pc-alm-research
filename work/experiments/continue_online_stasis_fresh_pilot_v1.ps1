param([Parameter(Mandatory=$true)][string]$Project)
$ErrorActionPreference = 'Stop'
$taskRoot = (Resolve-Path -LiteralPath $Project).Path
$taskPython = 'C:/Users/callofthenight/AppData/Local/Programs/Python/Python314/python.exe'
$taskRunner = Join-Path $taskRoot 'work/experiments/online_stasis_fresh_pilot_v1.py'
$taskTests = Join-Path $taskRoot 'work/experiments/test_online_stasis_fresh_pilot_v1.py'
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
& $taskPython -u $taskTests --project $taskRoot
if ($LASTEXITCODE -ne 0) { throw ("307 stopped at setup, exit={0}; no retry" -f $LASTEXITCODE) }
$taskPhases = @(
    @{ Stage='preflight'; Phase='predict' },
    @{ Stage='preflight'; Phase='score' },
    @{ Stage='pilot'; Phase='predict' },
    @{ Stage='pilot'; Phase='score' }
)
foreach ($taskStep in $taskPhases) {
    Write-Output ("307 stage={0} phase={1}" -f $taskStep.Stage,$taskStep.Phase)
    & $taskPython -u $taskRunner --project $taskRoot --stage $taskStep.Stage --phase $taskStep.Phase
    if ($LASTEXITCODE -ne 0) { throw ("307 stopped at {0}/{1}, exit={2}; evidence preserved, no retry" -f $taskStep.Stage,$taskStep.Phase,$LASTEXITCODE) }
}
