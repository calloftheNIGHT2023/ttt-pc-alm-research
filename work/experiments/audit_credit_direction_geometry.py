"""Frozen spectral geometry diagnosis, no query inputs or risk data."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def bound(cov,matrix):
    matrix=matrix/np.trace(matrix);values,vectors=np.linalg.eigh(cov);lam=values[-1];v=vectors[:,-1]
    delta=float(np.max(np.abs(values[:-1])));alpha=float(v@matrix@v);gamma=lam*lam*alpha;eta=2*lam*delta+delta*delta
    operator=cov@matrix@cov;actual_residual=operator-gamma*np.outer(v,v)
    residual_norm=float(np.linalg.norm(actual_residual,2))
    numerical_tolerance=64*np.finfo(float).eps*max(lam*lam,gamma,float(np.linalg.norm(operator,2)))
    assert residual_norm<=eta+numerical_tolerance
    _,vv=np.linalg.eigh(operator/np.max(np.abs(operator)));w=vv[:,-1]
    # Orthogonal projection avoids cancellation in sqrt(1-cosine^2).
    sine=float(np.linalg.norm(w-v*(v@w)))
    ceiling=min(1.,eta/(gamma-eta)) if gamma>eta else 1.
    assert sine<=ceiling+1e-9
    return dict(second_to_first_eigenvalue=float(max(0.,values[-2])/lam),alpha=alpha,actual_sine=sine,sine_ceiling=ceiling,
                informative=ceiling<1.,eta=float(eta),gamma=float(gamma),residual_norm=residual_norm,
                floating_psd_min_eigenvalue=float(values[0]),residual_check_tolerance=numerical_tolerance)


def verify():
    rng=np.random.default_rng(741362);rank_one=0;perturbed=0
    for _ in range(40):
        q,_=np.linalg.qr(rng.normal(size=(4,4)));z=rng.normal(size=(4,4));m=z@z.T
        cov=np.outer(q[:,0],q[:,0]);result=bound(cov,m);assert result['actual_sine']<1e-12;rank_one+=1
        for eps in [1e-6,1e-3,.1]:
            cov=q@np.diag([1.,eps,eps/2,eps/4])@q.T;bound(cov,m);perturbed+=1
    return dict(passed=True,rank_one_cases=rank_one,perturbation_bound_cases=perturbed)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--project',type=Path,required=True);args=parser.parse_args();root=args.project.resolve()
    inp=root/'results/credit_fiber/diagnostic';analysis=root/'results/credit_fiber/analysis';out=root/'results/credit_fiber/geometry'
    parent=json.loads((inp/'protocol.json').read_text());hashes=dict(parent['source_sha256'])
    for name,value in hashes.items():assert sha(Path(__file__).with_name(name))==value,name
    hashes[Path(__file__).name]=sha(Path(__file__))
    protocol=dict(source_sha256=hashes,parent_protocol_sha256=sha(inp/'protocol.json'),
                  input_audits_sha256=sha(inp/'audits.json'),geometry_rows_sha256=sha(analysis/'region_geometry.json'),seeds=parent['seeds'],
                  thresholds=[1e-6,1e-4,.01],sine_threshold=.01,verification=verify(),scope='all original cells; no queries, no risk selection')
    out.mkdir(parents=True,exist_ok=True);assert not (out/'protocol.json').exists();(out/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    audits=json.loads((inp/'audits.json').read_text());records=json.loads((analysis/'region_geometry.json').read_text())
    # Only geometric volume fields are consumed from derived analysis.
    volumes={(r['seed'],r['pattern']):r['volume'] for r in records};rows=[]
    for audit in audits:
        path=inp/audit['directions_file'];assert sha(path)==audit['directions_sha256']
        with np.load(path) as z:keys=z['keys'];covs=z['covariance'];matrix=z['scatter'][z['kinds'].tolist().index('augmented')]
        total=sum(volumes[audit['seed'],str(key)] for key in keys)
        for key,cov in zip(keys,covs):
            row=bound(cov,matrix);rows.append(dict(seed=audit['seed'],pattern=str(key),posterior_weight=volumes[audit['seed'],str(key)]/total,**row))
    summary=dict(regions=len(rows),informative_bound_regions=sum(r['informative'] for r in rows),
                 mean_task_weight_below_ratio={str(t):sum(r['posterior_weight'] for r in rows if r['second_to_first_eigenvalue']<=t)/16 for t in protocol['thresholds']},
                 mean_task_weight_with_sine_bound_001=sum(r['posterior_weight'] for r in rows if r['sine_ceiling']<=.01)/16,
                 observed_mean_task_weighted_sine=sum(r['posterior_weight']*r['actual_sine'] for r in rows)/16,
                 maximum_actual_sine=max(r['actual_sine'] for r in rows))
    assert len(rows)==175
    for name,value in [('rows.json',rows),('summary.json',dict(complete=True,**summary))]:
        (out/name).write_text(json.dumps(value,indent=2),encoding='utf-8')
    print(json.dumps(dict(complete=True,**summary),indent=2))


if __name__=='__main__':main()
