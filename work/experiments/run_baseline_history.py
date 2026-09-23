"""All 700 regions against complete unchanged PC/Adam/no-dual histories."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import platform
import time
import numpy as np
import baseline_history_capture as history
import exact_credit_hull as hull
from run_credit_wall_budget import load_inputs
from run_exact_credit_hull import UNION

OLD={'pc_history_full':'pc_native','nodual_history_full':'nodual_native','adam60_history_full':'adam60_native','strong_history_union':'strong_union'}


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())


def inherited_proof(x,v,reg,bank,oldbank,oldrow):
    assert oldrow['status']=='positive';keys={a.tobytes():i for i,a in enumerate(bank)}
    mapping=np.array([keys[a.tobytes()] for a in oldbank]);weights=np.zeros(len(bank));np.add.at(weights,mapping,np.array(oldrow['weights']))
    credit=np.array(oldrow['credit']);proof=hull.joint.certify_mixture(x,v,reg,bank,weights,credit)
    assert proof['accepted'],'An old exact positive certificate must survive inclusion with this checked embedding'
    return dict(weights=weights.tolist(),credit=credit.tolist(),proof=proof,origin='Existing exact positive proof embedded after raw solver returned unknown')


def summarize(records,old_records):
    lookup={(r['seed'],r['method']):r for r in records};oldlookup={(r['seed'],r['method']):r for r in old_records};summaries=[];paired=[]
    for method in history.VARIANTS:
        rr=[r for r in records if r['method']==method];raw=Counter();effective=Counter();pair=Counter();transitions=Counter();separations=[]
        for r in rr:
            raw.update(r['raw_counts']);effective.update(r['effective_counts']);seed=r['seed'];native=oldlookup[seed,'alm_native']['status_by_pattern'];prior=oldlookup[seed,OLD[method]]['status_by_pattern']
            for key,status in r['status_by_pattern'].items():
                pair[native[key]+'__'+status]+=1;transitions[prior[key]+'__'+status]+=1
                if (native[key],status) in [('positive','nonpositive'),('nonpositive','positive')]:separations.append(dict(seed=seed,pattern=key,alm=native[key],comparator=status))
        summaries.append(dict(method=method,raw_counts=dict(raw),effective_counts=dict(effective),directions_min=min(r['directions'] for r in rr),directions_max=max(r['directions'] for r in rr),
            mean_bank_bytes=float(np.mean([r['bank_bytes'] for r in rr])),diagnostic_solver_seconds=sum(r['solver_seconds'] for r in rr),rational_preparation_seconds=sum(r['rational_bank_seconds'] for r in rr)))
        paired.append(dict(comparator=method,status_pairs=dict(pair),old_to_full=dict(transitions),strict_separations=separations))
    return summaries,paired


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);root=ap.parse_args().project.resolve();base=root/'results/baseline_history';out=base/'development'
    prim=read(base/'primitive/summary.json');assert prim['passed'];hashes=dict(prim['source_sha256'])
    for name,value in hashes.items():assert sha(Path(__file__).with_name(name))==value,name
    hashes[Path(__file__).name]=sha(Path(__file__));inputs,data_hashes=load_inputs(root);prior=root/'results/exact_credit_hull/development';old_records=read(prior/'banks.json')
    protocol=dict(source_sha256=hashes,primitive_sha256=sha(base/'primitive/summary.json'),design_sha256=sha(root/'outputs/ttt-pc-alm-research/237_full_baseline_history_protocol.md'),
        input_hashes=data_hashes,prior_banks_sha256=sha(prior/'banks.json'),prior_summary_sha256=sha(prior/'summary.json'),seeds=sorted(inputs),methods=history.VARIANTS,
        union_components=history.LEARNERS,total_regions=700,total_method_regions=2800,queries_used=False,zero_removal='strict all-zero only',
        inherited_positive='Used only after a raw unknown result; embedded old proof is re-certified exactly, raw classification separately retained',
        threads={k:os.environ.get(k) for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS']},platform=platform.platform(),python=platform.python_version(),
        scope='Read-only full-trajectory credit capacity; diagnostic collection/solver costs, not online performance')
    assert all(v=='1' for v in protocol['threads'].values());out.mkdir(parents=True,exist_ok=True);assert not (out/'protocol.json').exists()
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8');captures=[];records=[]
    for seed,(x,v,regs,oldbanks) in inputs.items():
        full={};fullrefs={};oldbanks['strong_union']=np.concatenate([oldbanks[m] for m in UNION])
        for learner in history.LEARNERS:
            reference=history.capture(x,v,learner,False);actual=history.capture(x,v,learner,True);checks=history.verify_pair(reference,actual);old,arrays,meta=actual
            method=learner+'_history_full';source=data_hashes[seed]['banks'][learner+'_native'];path=root/source['file'];assert sha(path)==source['sha256']
            with np.load(path) as z:
                for key,index in [('regs',2),('bank',3),('labels',4),('steps',5)]:assert z[key].tobytes()==old[index].tobytes(),(seed,learner,key)
            assert old[3].tobytes()==oldbanks[learner+'_native'].tobytes()
            filename=f'history_{seed}_{learner}.npz';target=out/filename;assert not target.exists();np.savez_compressed(target,**arrays)
            cap=dict(seed=seed,learner=learner,method=method,reference=reference[2],actual=meta,checks=checks,arrays_file=filename,arrays_sha256=sha(target),old_source=source)
            captures.append(cap);full[method]=arrays['bank'];fullrefs[method]=dict(file=filename,sha256=sha(target))
            print(json.dumps(dict(capture=learner,seed=seed,directions=len(arrays['bank']),events=checks['trajectory_events'],old_subset=True)),flush=True)
        full['strong_history_union']=np.concatenate([full[m+'_history_full'] for m in history.LEARNERS])
        fullrefs['strong_history_union']=dict(parts=[fullrefs[m+'_history_full'] for m in history.LEARNERS])
        for method in history.VARIANTS:
            bank=full[method];start=time.perf_counter();rb=hull.RationalBank(bank);prep=time.perf_counter()-start;rows=[];rawcounts=Counter();counts=Counter();statuses={}
            priorrows=read(prior/f'bank_{seed}_{OLD[method]}.json');start=time.perf_counter()
            for i,reg in enumerate(regs):
                oldrow=priorrows[i];key=reg.tobytes().hex();assert oldrow['pattern']==key;oldresult=oldrow['result']
                result=hull.solve(x,v,reg,rb);status=result['status'];inherited=None;rawcounts[status]+=1
                if oldresult['status']=='positive':
                    assert status!='nonpositive',(seed,method,i,'positive cannot become nonpositive under inclusion')
                    if status=='unknown':inherited=inherited_proof(x,v,reg,bank,oldbanks[OLD[method]],oldresult);status='positive'
                rows.append(dict(index=i,pattern=key,result=result,effective_status=status,inherited=inherited,prior_status=oldresult['status']))
                counts[status]+=1;statuses[key]=status
            seconds=time.perf_counter()-start;filename=f'bank_{seed}_{method}.json';target=out/filename;assert not target.exists();target.write_text(json.dumps(rows,indent=2),encoding='utf-8')
            rec=dict(seed=seed,method=method,regions=len(regs),directions=len(bank),bank_bytes=bank.nbytes,bank_sha256=hashlib.sha256(bank.tobytes()).hexdigest(),bank_source=fullrefs[method],
                rational_bank_seconds=prep,solver_seconds=seconds,raw_counts=dict(rawcounts),effective_counts=dict(counts),status_by_pattern=statuses,rows_file=filename,rows_sha256=sha(target))
            records.append(rec);print(json.dumps({k:v for k,v in rec.items() if k not in ['status_by_pattern','rows_sha256','bank_sha256','bank_source']}),flush=True)
        # Checkpoint manifests are atomically replaceable diagnostics, not complete claims.
        (out/'captures.json').write_text(json.dumps(captures,indent=2),encoding='utf-8');(out/'banks.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
    summaries,paired=summarize(records,old_records);(out/'paired.json').write_text(json.dumps(paired,indent=2),encoding='utf-8')
    result=dict(execution_complete=True,captures=len(captures),banks=len(records),regions=700,total_method_regions=2800,summaries=summaries,
        comparisons=[{k:v for k,v in r.items() if k!='strict_separations'} for r in paired],source_sha256=sha(Path(__file__)),
        input_sha256={name:sha(out/name) for name in ['protocol.json','captures.json','banks.json','paired.json']},scope=protocol['scope'])
    (out/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
