"""397 formal entry: additionally bind the already-verified official dependencies.

Scientific calculation, registry and worker are unchanged from the passed
168-call preflight. This wrapper only strengthens frozen dependency coverage.
"""
from pathlib import Path
import os
import subprocess
import traceback
import run_prefix_deadline_pilot_v1 as original

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/original.io.BASE/'development_v1'
original_dependencies=original.io.dependencies


def dependencies(root):
    value=original_dependencies(root)
    trained=root/'results/matched_official_ttt/training_v1/protocol.json'
    prior=original.io.read(trained)
    for name,digest in prior['source_sha256'].items():
        assert original.io.sha(root/name)==digest,name
        if name in value['source_sha256']:
            assert value['source_sha256'][name]==digest
        value['source_sha256'][name]=digest
    official=root/'work/third_party/ttt-lm-pytorch'
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=official,text=True).strip()
    assert commit==prior['official_commit']=='cd831db10c8c9a0f6340f02da5613316a8a92b67'
    value.update(official_commit=commit,trained_dependency_hashes_explicitly_bound=True,
        scientific_code_unchanged_from_passed_prefix_preflight=True)
    return value


if __name__=='__main__':
    assert os.environ.get('OPENBLAS_NUM_THREADS')==os.environ.get('OMP_NUM_THREADS')=='1'
    assert original.io.read(ROOT/original.io.BASE/'audit_preflight_v1/summary.json')['passed']
    original.io.dependencies=dependencies
    OUT.mkdir(parents=True,exist_ok=False)
    try:
        original.run(ROOT,OUT,'run')
    except Exception:
        original.io.save(OUT/'failure.json',dict(traceback=traceback.format_exc(),automatic_retry=False));raise
