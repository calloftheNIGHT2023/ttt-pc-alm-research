"""413 exact component checks before actual support diagnostics."""
from fractions import Fraction as F
from itertools import product
from pathlib import Path
import time
import traceback
import numpy as np
import partial_support_certificate_v1 as core
import test_certified_partial_readout_v1 as old_test
import deadline_risk_io_v1 as io

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/partial_support_certificate/preflight_v1'
DESIGN='outputs/ttt-pc-alm-research/413_partial_support_certificate_protocol_v1.md'


def main():
    begin=time.perf_counter();rng=np.random.default_rng(413731)
    files=[Path(__file__),Path(core.__file__),Path(core.mathcore.__file__),ROOT/DESIGN]
    hashes={p.relative_to(ROOT).as_posix():io.sha(p) for p in files}
    counts=dict(ranks=0,volume_bounds=0,forest_completions=0,lipschitz_predictions=0,cone_transforms=0)
    assert core.exact_rank([])==0 and core.exact_rank([[0,0]])==0
    for _ in range(32):
        matrix=rng.integers(-3,4,(4,4)).tolist()
        rank=core.exact_rank(matrix)
        assert (rank==4)==bool(old_test.det_permutation(matrix));counts['ranks']+=1
    box_a,box_r=core.classifier.box_rows(4)
    assert core.cell_upper(box_a,box_r)['volume']==core.PRIOR;counts['volume_bounds']+=1
    # A small rotated box has a known exact determinant and exact volume.
    matrix=[[1,1,0,0],[1,-1,0,0],[0,0,1,0],[0,0,0,1]]
    rows=box_a+[list(map(F,row)) for row in matrix]+[[-F(t) for t in row] for row in matrix]
    rhs=box_r+[F(1,100)]*8
    bound=core.cell_upper(rows,rhs)
    expected=(F(2,100)**4)/2
    assert bound['volume']==expected;counts['volume_bounds']+=1
    inconsistent=core.cell_upper(box_a+[[F(0)]*4],box_r+[-F(1)])
    assert inconsistent['volume']==0;counts['volume_bounds']+=1
    inconsistent=core.cell_upper(box_a+[[F(1),F(0),F(0),F(0)]],box_r+[-F(1)])
    assert inconsistent['volume']==0;counts['volume_bounds']+=1
    parents=np.array([[[i],[1],[2],[1]] for i in range(3)],np.uint8)
    language=np.array([[1,1,1,j] for j in range(4)],np.uint8)
    full=np.array([[[0,1,1],[1,1,1],[2,1,1],[1,0,1]]],np.uint8)
    arrays=dict(support_order=np.arange(3),regions=full,pending_0=parents,language_1=language,
        x_observed=np.array([.1,.2,.3]),v_observed=np.array([.2,.3,.4]))
    metadata=dict(pending=[dict(k=2,start=5)])
    cover=core.forest_cover(arrays,metadata)
    for parent_id in range(3):
        for child_id,path in enumerate(language):
            if parent_id*len(language)+child_id<5:continue
            for suffix in language:
                candidate=np.concatenate([parents[parent_id],path[:,None],suffix[:,None]],axis=1)
                token=candidate.T.copy().tobytes()
                assert any(token.startswith(t) for t in cover);counts['forest_completions']+=1
    assert full[0].T.copy().tobytes() in cover
    assert core.forest_upper(arrays,metadata,max_cells=1)['volume']==core.PRIOR
    root_arrays=dict(arrays,pending_0=np.empty((1,4,0),np.uint8),language_0=language)
    root_meta=dict(pending=[dict(k=1,start=1)])
    assert core.forest_cover(root_arrays,root_meta)==[b'']
    assert core.forest_upper(root_arrays,root_meta)['reason']=='root_cover'
    vertices=np.array(list(product([-.1,.1],repeat=4)))
    indices=old_test.cube_faces(tuple(product([-1,1],repeat=4)))
    poly=dict(center=np.zeros(4),scale=np.ones(4),interior=np.zeros(4),facets=vertices[np.array(indices)])
    cones,proposal=core.cone_proposal(poly,box_a,box_r)
    safety=F(1048575,1048576)
    assert cones['mass']==(2*F(.1))**4*safety**4/6
    assert len(cones['simplices'])==8;counts['cone_transforms']+=1
    for _ in range(32):
        x=list(map(F,rng.random(6)));b=list(map(F,rng.uniform(-.12,.12,4)));q=list(map(F,rng.random(16)))
        v=[core.mathcore.forward(xx,b)+F(int(rng.integers(-1,2)))*core.EPS/2 for xx in x]
        intervals=core.lipschitz_intervals(x,v,q)
        for qq,(low,high) in zip(q,intervals):
            assert low<=core.mathcore.forward(qq,b)<=high;counts['lipschitz_predictions']+=1
    io.verify_hashes(ROOT,hashes)
    summary=dict(passed=True,counts=counts,source_sha256=hashes,seconds=time.perf_counter()-begin,
        actual_support_diagnostic_run=False,query_risk_compared=False)
    io.save(OUT/'summary.json',summary);print(summary,flush=True)


if __name__=='__main__':
    gate=io.read(ROOT/'results/certified_partial_readout/math_preflight_v1/summary.json')
    assert gate['passed'];io.verify_hashes(ROOT,gate['source_sha256'])
    OUT.mkdir(parents=True,exist_ok=False)
    try:main()
    except Exception:
        io.save(OUT/'failure.json',dict(traceback=traceback.format_exc(),automatic_retry=False));raise
