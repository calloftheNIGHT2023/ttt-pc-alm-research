"""Serialization-only successor of the preserved geometry diagnostic."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import audit_credit_direction_geometry as original


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def native(value):
    if isinstance(value,np.generic):return value.item()
    raise TypeError(type(value).__name__)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--project',type=Path,required=True);args=parser.parse_args();root=args.project.resolve()
    inp=root/'results/credit_fiber/diagnostic';analysis=root/'results/credit_fiber/analysis';old=root/'results/credit_fiber/geometry';out=root/'results/credit_fiber/geometry_v2'
    failed=json.loads((old/'protocol.json').read_text());hashes=dict(failed['source_sha256'])
    for name,value in hashes.items():assert sha(Path(__file__).with_name(name))==value,name
    hashes[Path(__file__).name]=sha(Path(__file__))
    protocol=dict(failed,source_sha256=hashes,failed_protocol_sha256=sha(old/'protocol.json'),
                  failed_run='preserved NumPy bool serialization failure; numerical bound unchanged',verification=original.verify())
    assert sha(inp/'audits.json')==protocol['input_audits_sha256'] and sha(analysis/'region_geometry.json')==protocol['geometry_rows_sha256']
    out.mkdir(parents=True,exist_ok=True);assert not (out/'protocol.json').exists()
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2,default=native),encoding='utf-8')
    audits=json.loads((inp/'audits.json').read_text());records=json.loads((analysis/'region_geometry.json').read_text())
    volumes={(r['seed'],r['pattern']):r['volume'] for r in records};rows=[]
    for audit in audits:
        path=inp/audit['directions_file'];assert sha(path)==audit['directions_sha256']
        with np.load(path) as z:keys=z['keys'];covs=z['covariance'];matrix=z['scatter'][z['kinds'].tolist().index('augmented')]
        total=sum(volumes[audit['seed'],str(key)] for key in keys)
        for key,cov in zip(keys,covs):
            row=original.bound(cov,matrix);rows.append(dict(seed=audit['seed'],pattern=str(key),posterior_weight=volumes[audit['seed'],str(key)]/total,**row))
    result=dict(complete=True,regions=len(rows),informative_bound_regions=sum(r['informative'] for r in rows),
                mean_task_weight_below_ratio={str(t):sum(r['posterior_weight'] for r in rows if r['second_to_first_eigenvalue']<=t)/16 for t in protocol['thresholds']},
                mean_task_weight_with_sine_bound_001=sum(r['posterior_weight'] for r in rows if r['sine_ceiling']<=.01)/16,
                observed_mean_task_weighted_sine=sum(r['posterior_weight']*r['actual_sine'] for r in rows)/16,
                maximum_actual_sine=max(r['actual_sine'] for r in rows),scope=protocol['scope'])
    assert len(rows)==175
    for name,value in [('rows.json',rows),('summary.json',result)]:
        (out/name).write_text(json.dumps(value,indent=2,default=native),encoding='utf-8')
    print(json.dumps(result,indent=2,default=native))


if __name__=='__main__':main()
