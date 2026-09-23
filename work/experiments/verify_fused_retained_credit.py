"""All-task equivalence of fused online retention with frozen diagnostics."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import retained_credit_memory as model
import retained_credit_capture as diagnostic


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--project',type=Path,required=True);args=parser.parse_args();root=args.project/'results';parentdir=root/'retained_credit/cross_bank';out=root/'retained_credit/fused_verification'
    parent=json.loads((parentdir/'protocol.json').read_text());hashes=parent['source_sha256'].copy()
    for name,h in hashes.items():assert sha(Path(__file__).with_name(name))==h,name
    for name in [Path(__file__).name,Path(model.__file__).name]:hashes[name]=sha(Path(__file__).with_name(name))
    original=json.loads((root/'budgeted_credit_memory/development/protocol.json').read_text());cfgs={c['name']:c for c in original['configs']}
    cross={(r['seed'],r['method']):r for r in json.loads((parentdir/'rows.json').read_text()) if r['bank']=='retained'};matched={(r['seed'],r['method']):r for r in json.loads((root/'retained_credit/diagnostic/rows.json').read_text())}
    out.mkdir(parents=True,exist_ok=True);assert not (out/'protocol.json').exists()
    protocol=dict(source_sha256=hashes,seeds=parent['seeds'],profiles=parent['profiles'],primitive_verification=model.verify(),scope='all frozen diagnostics reproduced before actual online timing; diagnostic trace/capture costs not benchmark')
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8');rows=[]
    for seed in protocol['seeds']:
        rng=np.random.default_rng(seed);truth=rng.uniform(-.12,.12,4);xx=rng.uniform(0,1,24);x=xx[:4];v=(model.base.forward(xx,truth)+np.random.default_rng(seed+19000000).uniform(-model.base.EPS,model.base.EPS,24))[:4]
        for name in protocol['profiles']:
            cfg=dict(**cfgs[name],retention_mode='bank');before=diagnostic.capture(x,v,cfg,True);begin=time.perf_counter();bank,regs,meta,collectors=model.prepare(x,v,cfg,True);elapsed=time.perf_counter()-begin
            assert np.array_equal(before[0],bank);assert meta['pre_retention_pattern_keys']==matched[seed,name]['before_keys'];assert meta['post_matching_pattern_keys']==matched[seed,name]['after_keys'];assert meta['post_retention_pattern_keys']==cross[seed,name]['after_keys']
            pending=0;retained=0;traces=[]
            for a,b in zip(collectors,before[3]):
                assert a.trajectory.hexdigest()==b.trajectory.hexdigest();assert a.forward==b.forward and a.split==b.split and list(a.pending)==list(b.pending) and set(a.retained)==set(b.retained)
                for key in a.pending:assert a.pending[key][0]==b.pending[key][0] and np.array_equal(a.pending[key][1],b.pending[key][1]) and a.pending[key][2]==b.pending[key][2]
                for key,item in a.retained.items():
                    other=b.retained[key];assert np.array_equal(item['a'],other['a']);assert {k:v for k,v in item.items() if k!='a'}=={k:v for k,v in other.items() if k!='a'}
                pending+=len(a.pending);retained+=len(a.retained);traces.append(a.trajectory.hexdigest())
            rows.append(dict(seed=seed,method=name,trajectory_sha256=traces,original_pending_records=pending,retained_records=retained,bank_and_three_sequences_bitwise=True,
                diagnostic_fused_prepare_seconds=elapsed,retention_bookkeeping_seconds=meta['retention_bookkeeping_seconds'],retained_numeric_bytes=meta['retained_numeric_bytes']))
        (out/'rows.json').write_text(json.dumps(rows,indent=2),encoding='utf-8');print(json.dumps(dict(seed=seed,completed=len(rows)//9,cases=len(rows))),flush=True)
    summary=dict(source_hashes=len(hashes),cases=len(rows),original_pending_records=sum(r['original_pending_records'] for r in rows),retained_records=sum(r['retained_records'] for r in rows),
        summary=[dict(method=n,mean_diagnostic_fused_prepare_seconds=float(np.mean([r['diagnostic_fused_prepare_seconds'] for r in rows if r['method']==n])),mean_retention_bookkeeping_seconds=float(np.mean([r['retention_bookkeeping_seconds'] for r in rows if r['method']==n]))) for n in protocol['profiles']],scope=protocol['scope'])
    (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8');print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
