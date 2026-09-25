"""Evaluator-only structural diagnosis; no new predictions or model selection."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import vector_interval_memory as m


def structure(point,x,weights):
    d,w=point.shape;h=x.copy();jac=np.zeros((len(x),w,d*w));dead=[]
    for layer,weight in enumerate(weights):
        z=h@weight.T+point[layer];derivative=m.family.deriv(z)
        dead.append(np.all(derivative==0,axis=0).tolist())
        jac=np.einsum('ab,nbp->nap',weight,jac,optimize=True)
        jac[:,np.arange(w),layer*w+np.arange(w)]+=1
        jac*=derivative[:,:,None];h=m.family.activation(z)
    sv=np.linalg.svd(jac.reshape(-1,d*w),compute_uv=False)
    return {'context_jacobian_rank':int(np.sum(sv>1e-9)),'smallest_singular_value':float(sv[-1]),
        'inactive_all_context_by_layer':dead,'inactive_all_context_counts':[sum(a) for a in dead]}


def main():
    p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True);args=p.parse_args()
    proto=json.loads((args.results/'protocol.json').read_text());rows=json.loads((args.results/'episodes.json').read_text());records=[]
    weights=m.family.make_weights(proto['depth'],proto['width'])
    for seed in range(proto['seed0'],proto['seed0']+proto['count']):
        rng=np.random.default_rng(seed);truth=rng.uniform(-m.PRIOR,m.PRIOR,(proto['depth'],proto['width']));x=rng.uniform(-1,1,(max(proto['stages']),proto['width']))
        teacher=structure(truth,x[:8],weights)
        for name in ['input_exact240','adam240_r16','lbfgs64']:
            row=next(r for r in rows if r['seed']==seed and r['n_context']==8 and r['method']==name)
            point=np.array(row['anchor_output'])
            records.append({'seed':seed,'method':name,'query_mse_evaluator_only':row['query_mse'],'support_max_error':row['support_max_error'],
                'outside_known_prior_coordinates':int(np.sum(np.abs(point)>m.PRIOR+1e-10)),
                'max_prior_box_excess':float(np.maximum(np.abs(point)-m.PRIOR,0).max()),'parameter_l2_error_evaluator_only':float(np.linalg.norm(point-truth)),
                'teacher_structure_evaluator_only':teacher,'fitted_structure':structure(point,x[:8],weights)})
    summary={name:{'outside_prior_streams':sum(r['outside_known_prior_coordinates']>0 for r in records if r['method']==name),
        'fitted_rank_deficient_streams':sum(r['fitted_structure']['context_jacobian_rank']<proto['depth']*proto['width'] for r in records if r['method']==name)} for name in ['input_exact240','adam240_r16','lbfgs64']}
    result={'records':records,'summary':summary,'scope':'rank is local identifiability only; full rank does not prove global identifiability; teacher quantities evaluator-only',
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    out=args.results.parent/(args.results.name+'_analysis');out.mkdir(exist_ok=True);(out/'identifiability.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({'summary':summary,'large_error_records':[r for r in records if r['query_mse_evaluator_only']>.001]}),flush=True)


if __name__=='__main__':main()
