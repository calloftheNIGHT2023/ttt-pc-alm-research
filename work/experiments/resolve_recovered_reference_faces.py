"""Resolve numerical zero-width reference cells by exact opposing constraints.

No small-volume threshold: a rational identity proves an affine hyperplane
or infeasibility. Original unresolved results are retained without edits.
"""
import argparse
import hashlib
import json
from fractions import Fraction as F
from pathlib import Path
import numpy as np
import enumerate_support_modes as reference


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def certify(x,v,reg):
    aa,rr=reference.constraints(x,v,reg,True);d=reg.shape[0]
    for j in range(d):
        vec=[F(0)]*d;vec[j]=F(1);aa.extend([vec,[-t for t in vec]]);rr.extend([F(.12),F(.12)])
    for i,a in enumerate(aa):
        nonzero=[j for j,z in enumerate(a) if z]
        if not nonzero:continue
        j=nonzero[0]
        for k in range(i+1,len(aa)):
            if aa[k][j]==0:continue
            scale=-a[j]/aa[k][j]
            if scale<=0 or any(a[t]+scale*aa[k][t] for t in range(d)):continue
            gap=rr[i]+scale*rr[k]
            if gap>0:continue
            return dict(first_row=i,second_row=k,positive_scale=str(scale),rhs_sum=str(gap),
                normal=list(map(str,a)),rhs=str(rr[i]),
                kind='exact_affine_hyperplane_zero_volume' if gap==0 else 'exact_opposite_infeasible',
                justification='Nonzero normal and positive combination gives equality or contradiction; zero Lebesgue volume under continuous uniform prior')
    return None


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);root=ap.parse_args().project.resolve()
    inp=root/'results/recovered_support_reference/development';online=root/'results/recovered_online_comparison/development'
    p=json.loads((inp/'protocol.json').read_text());assert json.loads((inp.parent/'analysis/summary.json').read_text())['passed']
    hashes=dict(p['source_sha256'])
    for name,value in hashes.items():assert sha(Path(__file__).with_name(name))==value,name
    hashes[Path(__file__).name]=sha(Path(__file__))
    out=inp.parent/'exact_faces';out.mkdir(parents=True,exist_ok=True);assert not (out/'protocol.json').exists()
    protocol=dict(source_sha256=hashes,parent_protocol_sha256=sha(inp/'protocol.json'),
        rule='Only exact nonzero opposing constraint rows with nonpositive rhs sum; no rounding tolerance',
        scope='Offline resolution of every unresolved reference cell; unchanged original files; continuous-prior volume only')
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8');rows=[];proofs=[]
    for row in json.loads((inp/'coverage.json').read_text()):
        file=inp/row['reference_file'];assert sha(file)==row['reference_sha256'];ref=json.loads(file.read_text())
        x=np.array(ref['x']);v=np.array(ref['v']);unresolved=[];count=0
        for item in ref['final_geometry_unresolved']:
            reg=np.frombuffer(bytes.fromhex(item['pattern']),np.uint8).reshape(4,4);proof=certify(x,v,reg)
            if proof is None:unresolved.append(item)
            else:proofs.append(dict(seed=row['seed'],pattern=item['pattern'],original_reason=item['reason'],**proof));count+=1
        complete=ref['enumeration_completed'] and not unresolved;coverage=[]
        if complete:
            polys={r['pattern']:r for r in ref['positive_regions']};total=sum(r['volume'] for r in polys.values())
            for cfg in p['configs']:
                detail=online/f'detail_{row["seed"]}_{cfg["name"]}.json';meta=json.loads(detail.read_text());found=set(meta['positive_mode_keys'])
                assert found<=set(polys)
                coverage.append(dict(method=cfg['name'],detail_file=detail.name,detail_sha256=sha(detail),found_patterns=sorted(found),
                    missing_patterns=sorted(set(polys)-found),mass_fraction=sum(polys[k]['volume'] for k in found)/total))
        rows.append(dict(seed=row['seed'],reference_file=row['reference_file'],reference_sha256=row['reference_sha256'],complete=complete,
            certified_zero_or_empty=count,remaining_unresolved=unresolved,coverage=coverage))
    summaries=[]
    for cfg in p['configs']:
        pairs=[(r['seed'],next(c for c in r['coverage'] if c['method']==cfg['name'])) for r in rows if r['complete']]
        summaries.append(dict(method=cfg['name'],complete_reference_tasks=len(pairs),
            mean_mass_fraction=float(np.mean([r['mass_fraction'] for _,r in pairs])) if len(pairs)==len(rows) else None,
            minimum_mass_fraction=min(r['mass_fraction'] for _,r in pairs) if len(pairs)==len(rows) else None,
            all_regions_found_tasks=sum(not r['missing_patterns'] for _,r in pairs),
            missing=[dict(seed=s,mass_fraction=r['mass_fraction'],patterns=r['missing_patterns']) for s,r in pairs if r['missing_patterns']]))
    result=dict(execution_complete=True,tasks=len(rows),complete_references=sum(r['complete'] for r in rows),exact_face_proofs=len(proofs),
        source_hashes=len(hashes),summaries=summaries,scope=protocol['scope'])
    for name,value in [('coverage.json',rows),('proofs.json',proofs),('summary.json',result)]:
        (out/name).write_text(json.dumps(value,indent=2),encoding='utf-8')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
