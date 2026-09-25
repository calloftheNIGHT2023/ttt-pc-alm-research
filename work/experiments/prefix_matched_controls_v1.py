"""395 same-cohort prefix models and task-local anytime optimizer portfolios."""
from pathlib import Path
import hashlib
import json
import time
import numpy as np

STAGES=(4,8,16,24)
PORTFOLIOS={
    'adam_gn64_anytime': [('adam',240,64),('gauss_newton',40,64)],
    'adam_gn64_256_anytime': [('adam',240,64),('gauss_newton',40,64),('adam',240,256),('gauss_newton',40,256)]}


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def read(p):
    return json.loads(p.read_text(encoding='utf-8'))


def require_training(root):
    folder=root/'results/matched_official_ttt'
    audit=read(folder/'training_audit_v1/summary.json')
    assert audit['passed'] and audit['training_summary_sha256']==sha(folder/'training_v1/summary.json')
    summary=read(folder/'training_v1/summary.json'); assert summary['passed'] and summary['trained_models']==8
    for name,digest in summary['outputs_sha256'].items():
        assert sha(folder/'training_v1'/name)==digest
    return folder/'training_v1'


def cohort_configs(root):
    folder=require_training(root)
    return [dict(name=c['name'],kind='cohort',training_config=c) for c in read(folder/'protocol.json')['configs']]


def load_model(config,root):
    import torch
    import matched_official_ttt_v1 as official
    import matched_shifted_meta_models as population
    torch.set_num_threads(1)
    folder=require_training(root);cfg=config['training_config'];name=cfg['name']
    frozen=next(c for c in read(folder/'protocol.json')['configs'] if c['name']==name)
    assert cfg==frozen
    metadata=next(r for r in read(folder/'training_summary.json') if r['method']==name)
    checkpoint=folder/(name+'.pt');assert sha(checkpoint)==metadata['checkpoint_sha256']
    if cfg['kind']=='official_ttt':
        model=official.MatchedOfficialTTT(cfg['head_dim'],cfg['inner_passes'],cfg['key_basis'],cfg['rate_mode']).double()
    else:
        model=population.make_model(cfg)
    model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True));model.eval()
    return model,dict(checkpoint_sha256=sha(checkpoint),training_config=cfg,training_manifest=metadata,
        named_shared_model_bytes=sum(t.numel()*t.element_size() for t in model.state_dict().values()))


def fit_model(model,x,v,q):
    import torch
    assert x.shape==v.shape and x.ndim==q.ndim==1 and len(x) in STAGES
    tick=time.perf_counter()
    xx,vv,qq=[torch.from_numpy(np.ascontiguousarray(a,dtype=np.float64))[None] for a in (x,v,q)]
    state=None;stages=[n for n in STAGES if n<=len(x)]
    with torch.no_grad():
        for n in stages:
            state=model.adapt(xx[:,:n],vv[:,:n],state)
        raw=model.predict(state,qq)[0].detach().cpu().numpy().copy()
    tensors=state if isinstance(state,dict) else ({'value':state} if isinstance(state,torch.Tensor) else {str(i):t for i,t in enumerate(state)})
    arrays={'fast_'+name:t.detach().cpu().numpy().copy() for name,t in tensors.items()}
    arrays.update(x=x.copy(),v=v.copy(),q=q.copy(),raw_prediction=raw,prediction=np.clip(raw,0.,1.))
    metadata=dict(stages=stages,support_exposures=sum(stages),warm_trajectory=True,
        named_fast_state_bytes=sum(t.numel()*t.element_size() for t in tensors.values()),
        returned_array_bytes=sum(a.nbytes for a in arrays.values()),query_targets_accessed=False,
        total_seconds=time.perf_counter()-tick)
    return arrays,metadata


def fit_portfolio(x,v,q,*,seed,name,emit=None):
    import n24_optimizer_controls_v1 as optimizer
    import candidate_set_readout_v1 as reader
    assert name in PORTFOLIOS and len(x) in STAGES
    begin=time.perf_counter();polys={};classified={};arrays={};records=[]
    best_point=None;best_error=float('inf');prediction=None
    for stage,(family,steps,restarts) in enumerate(PORTFOLIOS[name]):
        a,m=optimizer.fit(x,v,q,seed=seed,family=family,steps=steps,restarts=restarts,readout='point')
        point=a['selected_point'];error=float(np.max(abs(optimizer.old.base.forward(x,point)-v)))
        if error<best_error:
            best_error=error;best_point=point.copy()
        keys=sorted({optimizer.old.base.pattern(x,b).astype(np.uint8).tobytes().hex() for b in a['best_bank']})
        new_keys=[k for k in keys if k not in classified];tick=time.perf_counter()
        for key in new_keys:
            poly,note=reader.local.explicit_geometry(x,v,key);classified[key]=note
            if poly is not None:
                assert note['positive_volume_certified'] and note['numerical_volume_available']
                polys[key]=poly
        geometry_seconds=time.perf_counter()-tick;accepted=sorted(polys);tick=time.perf_counter()
        if accepted:
            points,allocation=reader.shared.draw(polys,accepted,2048,np.random.default_rng(np.random.SeedSequence([249911,seed,2048])))
            prediction=np.clip(reader.shared.geometry.make_predict(points)(q),0.,1.)
        else:
            points=np.empty((0,4));allocation=np.empty(0,int)
            prediction=np.clip(optimizer.old.base.forward(q,best_point),0.,1.)
        reading_seconds=time.perf_counter()-tick
        if emit is not None:
            emit('fallback',prediction)
        arrays.update({f'stage_{stage}_'+k:value for k,value in a.items()})
        arrays.update({f'stage_{stage}_points':points,f'stage_{stage}_allocation':allocation,
                       f'stage_{stage}_prediction':prediction.copy()})
        records.append(dict(stage=stage,family=family,steps=steps,restarts=restarts,optimizer=m,
            current_member_keys=keys,new_geometry_keys=new_keys,cumulative_keys=sorted(classified),
            positive_keys=accepted,geometry_seconds=geometry_seconds,reading_seconds=reading_seconds,
            current_best_support_error=best_error,cumulative_seconds=time.perf_counter()-begin,
            retained_geometry_numeric_bytes=sum(t.nbytes for poly in polys.values() for t in poly.values() if isinstance(t,np.ndarray))))
    arrays.update(x=x.copy(),v=v.copy(),q=q.copy(),prediction=prediction.copy())
    metadata=dict(name=name,stages=records,classified=reader.serializable(classified),
        final_positive_modes=sorted(polys),total_seconds=time.perf_counter()-begin,
        returned_array_bytes=sum(a.nbytes for a in arrays.values()),
        query_targets_accessed=False,global_bp_used=True,complete_posterior_claim=False,
        task_local_geometry_reuse_only=True,cross_method_or_task_cache=False,
        cumulative_intermediate_packets=True,intermediate_packet_kind='fallback',
        all_discovery_geometry_and_readout_costs_included=True)
    return arrays,metadata
