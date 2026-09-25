"""306 support-only independent audit of the singleton norm reduction fix."""
import argparse
from pathlib import Path
import numpy as np
from diagnose_gradient_flat_split_states_v1 import residuals,read,save,sha


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--project',type=Path,required=True)
    root=ap.parse_args().project.resolve();out=root/'results/online_stasis_controls/reduction_audit_v1'
    out.mkdir(parents=True,exist_ok=False)
    source=root/'results/observable_trap_continuation/development_v1'
    files=read(source/'before_reference_manifest.json')['files_sha256'];rows=[]
    for seed in range(5910000,5910064):
        name=f'{seed}_proposals.npz';assert sha(source/name)==files[name]
        with np.load(source/name,allow_pickle=False) as z:
            mask=z['mask_forward_stasis']
            for kind,whole in [('multiplier',z['initial_u']),('residual',residuals(z['initial_b'],z['initial_h'],z['x_observed']))]:
                expected=np.sum(whole*whole,axis=(0,2))[mask];selected=whole[:,mask].copy()
                nested=np.sum(np.sum(selected*selected,axis=2),axis=0)
                padded=np.concatenate([selected,np.zeros((4,2,4))],axis=1)
                alternative=np.sum(padded*padded,axis=(0,2))[:len(expected)]
                assert nested.tobytes()==expected.tobytes()==alternative.tobytes()
                scalar=np.array([sum(float(v*v) for v in selected[:,i,:].ravel()) for i in range(len(expected))])
                rows.append(dict(seed=seed,kind=kind,states=len(expected),passed=True,
                    scalar_unequal=int(np.sum(scalar!=expected)),reference_sha256=files[name]))
    save(out/'rows.json',rows)
    save(out/'summary.json',dict(passed=True,norms=sum(r['states'] for r in rows),tasks=64,
        scalar_unequal=sum(r['scalar_unequal'] for r in rows),nested_and_padded_bitwise_equal=True,
        query_targets_accessed=False,source_sha256={Path(__file__).name:sha(Path(__file__))},
        outputs_sha256={'rows.json':sha(out/'rows.json')}))
    print(dict(passed=True,norms=sum(r['states'] for r in rows),scalar_unequal=sum(r['scalar_unequal'] for r in rows)),flush=True)
