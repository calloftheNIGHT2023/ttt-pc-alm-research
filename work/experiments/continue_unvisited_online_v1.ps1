$ErrorActionPreference = 'Stop'
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
$researchPython = 'C:\Users\callofthenight\AppData\Local\Programs\Python\Python314\python.exe'
& $researchPython work/experiments/run_unvisited_online_v1.py --stage preflight
if ($LASTEXITCODE -ne 0) { throw '353 preflight failed' }
& $researchPython work/experiments/run_unvisited_online_v1.py --stage development
if ($LASTEXITCODE -ne 0) { throw '353 development failed' }
& $researchPython work/experiments/evaluate_unvisited_online_v1.py
if ($LASTEXITCODE -ne 0) { throw '353 evaluation failed' }
& $researchPython work/experiments/audit_unvisited_online_v1.py
if ($LASTEXITCODE -ne 0) { throw '353 independent audit failed' }
