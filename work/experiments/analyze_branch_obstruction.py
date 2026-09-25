import argparse,hashlib,json
from pathlib import Path
import numpy as np
from audit_branch_obstruction import cell_lp
import vector_interval_memory as m

p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True);args=p.parse_args()
proto=json.loads((args.results/'protocol.json').read_text());data=json.loads((args.results/'diagnostics.json').read_text())
assert all(hashlib.sha256(Path(__file__).with_name(n).read_bytes()).hexdigest()==h for n,h in proto['source_sha256'].items())
assert len(data['rows'])==2*4*3*16*3
summary=[]
for width in proto['widths']:
    for n in proto['stages']:
        group=[r for r in data['rows'] if r['width']==width and r['n_context']==n]
        methods={name:{(r['seed'],r['restart']):r for r in group if r['method']==name} for name in proto['methods']}
        local=methods['local480'];adam=methods['adam240_010'];lbfgs=methods['lbfgs300']
        summary.append({'width':width,'n_context':n,'starts':len(local),
            'feasible':{name:sum(r['support_feasible'] for r in rows.values()) for name,rows in methods.items()},
            'local_feasible_adam_not':sum(local[k]['support_feasible'] and not adam[k]['support_feasible'] for k in local),
            'adam_feasible_local_not':sum(adam[k]['support_feasible'] and not local[k]['support_feasible'] for k in local),
            'local_feasible_lbfgs_not':sum(local[k]['support_feasible'] and not lbfgs[k]['support_feasible'] for k in local),
            'both_bp_infeasible_local_feasible':sum(local[k]['support_feasible'] and not adam[k]['support_feasible'] and not lbfgs[k]['support_feasible'] for k in local),
            'adam_infeasible_projected_gradient_below_1e6':sum(not r['support_feasible'] and r['projected_gradient_norm']<1e-6 for r in adam.values()),
            'local_lower_context_band_objective':sum(local[k]['band_objective']<adam[k]['band_objective'] for k in local)})
# Hand-solvable cell: both points forced onto the negative branch, min max error = .6.
x=np.array([[-.15],[.15]]);weights=np.ones((1,1,1));truth=np.array([[.2]])
v=m.forward(truth[None],x,weights)[0]
bad=cell_lp(np.array([[-.2]]),x,v,weights);good=cell_lp(truth,x,v,weights)
assert abs(bad['minimum_max_error']-.6)<1e-10
assert abs(bad['dual_box_lower_bound']-.6)<1e-10
assert good['minimum_max_error']<1e-10 and good['dual_box_lower_bound']<1e-10
out={'source_hashes_match':True,'summary':summary,'cell_cases':data['cells'],'lp_hand_audit':{'wrong_cell':bad,'true_cell':good},
    'scope':'paired-start development diagnostics, not a significance test or new method'}
(args.results/'analysis.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
print(json.dumps(out),flush=True)
