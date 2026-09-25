"""Compatibility adapter: old regression geometry lists may be explicit null."""
import argparse
from pathlib import Path
import traceback
import numpy as np
import run_prefix_language_join_v1 as original


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['preflight','development'],required=True);args=ap.parse_args()
    root=Path(__file__).resolve().parents[2];oldread=original.read;oldhashes=original.hashes
    def read(path):
        result=oldread(path)
        if Path(path).resolve()==root/'results/context_block_scaling/development/rows.json':
            result=[dict(r,positive_mode_keys=[]) if r.get('positive_mode_keys') is None else r for r in result]
        return result
    def hashes(project):
        result=oldhashes(project)
        for name in ['work/experiments/run_prefix_language_join_v2.py',
                     'outputs/ttt-pc-alm-research/368_null_metadata_repair_v2.md',
                     'results/prefix_language_join/development_v1/failure.json']:
            result[name]=original.sha(project/name)
        return result
    original.read=read;original.hashes=hashes;original.BASE='results/prefix_language_join_v2'
    out=root/original.BASE/(args.stage+'_v1');out.mkdir(parents=True,exist_ok=False)
    try:
        original.run(root,out,args.stage)
        if args.stage=='development':
            count=0;failed=root/'results/prefix_language_join/development_v1'
            for file in failed.glob('*/arrays.npz'):
                counterpart=out/file.parent.name/'arrays.npz'
                with np.load(file,allow_pickle=False) as a,np.load(counterpart,allow_pickle=False) as b:
                    count+=original.same({k:a[k] for k in a.files},{k:b[k] for k in b.files},list(a.files))
            original.save(out/'repair_replay.json',dict(passed=True,original_failed_call_arrays=count,
                failure_sha256=original.sha(failed/'failure.json'),scope='Only post-fit metadata reading changed'))
            print(dict(repair_replay_passed=True,arrays=count),flush=True)
    except Exception:original.save(out/'failure.json',dict(traceback=traceback.format_exc(),automatic_retry=False));raise


if __name__=='__main__':main()
