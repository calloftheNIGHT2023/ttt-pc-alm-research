$ErrorActionPreference = 'Stop'
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
$researchPython = 'C:\Users\callofthenight\AppData\Local\Programs\Python\Python314\python.exe'
& $researchPython work/experiments/run_support_language_online_v1.py --stage preflight
if ($LASTEXITCODE -ne 0) { throw '345 preflight failed' }
& $researchPython work/experiments/run_support_language_online_v1.py --stage development
if ($LASTEXITCODE -ne 0) { throw '345 development failed' }
& $researchPython work/experiments/evaluate_support_language_online_v1.py
if ($LASTEXITCODE -ne 0) { throw '345 evaluation failed' }
& $researchPython work/experiments/audit_support_language_online_v1.py
if ($LASTEXITCODE -ne 0) { throw '345 independent audit failed' }
