"""405 registry/dependency/route checks; reuse real worker tests, not new calls."""
from pathlib import Path
import itertools
import inspect
import time
import traceback
import runtime_matched_prefix_io_v1 as io
import prefix_deadline_worker_v1 as original
import prefix_deadline_worker_v2 as updated

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/io.BASE/'binding_preflight_v1'


def main():
    begin=time.perf_counter();value=io.dependencies(ROOT)
    configurations=io.configs(ROOT)
    assert configurations==io.previous.configs(ROOT) and len(configurations)==42
    assert io.PRIMARY=='regional_active512_dfs_farthest_x' and io.PRIMARY_BUDGET==.5
    assert io.SEEDS==[5920000,5920001,5920002,5920003]
    assert io.STAGES==[4,8,16,24] and io.BUDGETS==[.25,.5]
    first=[(ci,cfg['name'],seed,n,budget) for ci,cfg,jobs in io.jobs(configurations,io.SEEDS,io.STAGES,io.BUDGETS) for seed,n,budget in jobs]
    second=[(ci,cfg['name'],seed,n,budget) for ci,cfg,jobs in io.jobs(configurations,io.SEEDS,io.STAGES,io.BUDGETS) for seed,n,budget in jobs]
    assert first==second and len(first)==len(set(first))==1344
    expected={(i,c['name'],s,n,b) for i,c in enumerate(configurations) for s,n,b in itertools.product(io.SEEDS,io.STAGES,io.BUDGETS)}
    assert set(first)==expected
    assert updated.original is original
    assert updated.Session.run is original.Session.run and updated.Session.close is original.Session.close
    # Read and bind the actual entry, rather than treating matching names as evidence.
    source=inspect.getsource(updated.worker)
    assert source.index('import n24_optimizer_controls_v1')<source.index('original.worker(connection, config, root)')
    assert "config['kind'] == 'portfolio'" in source
    checked=io.read(ROOT/'results/portfolio_runtime_setup/preflight_v1/summary.json')
    assert checked['passed'] and checked['calls']==32 and checked['saved_arrays_replayed']==1120
    io.save(OUT/'jobs.json',first)
    io.save(OUT/'summary.json',dict(passed=True,registry_entries=42,deterministic_unique_jobs=1344,
        same_controller_run_and_cleanup=True,reused_worker_preflight_calls=32,new_worker_calls=0,
        source_sha256=value['source_sha256'],pretrained_sha256=value['pretrained_sha256'],
        reused_preflight_summary_sha256=value['reused_preflight_summary_sha256'],
        outputs_sha256={'jobs.json':io.sha(OUT/'jobs.json')},query_targets_accessed=False,
        seconds=time.perf_counter()-begin))
    print({k:v for k,v in io.read(OUT/'summary.json').items() if not k.endswith('sha256')},flush=True)


if __name__=='__main__':
    OUT.mkdir(parents=True,exist_ok=False)
    try:main()
    except Exception:
        io.save(OUT/'failure.json',dict(traceback=traceback.format_exc(),automatic_retry=False));raise
