"""Compose the byte-identical endpoint bound with interval verification."""
import numpy as np
import online_interval_h2 as original
import endpoint_factorized_credit_bank as endpoint

prior=original.prior
base=original.base
neighbor=original.neighbor


class EndpointIntervalBank(original.IntervalBank):
    layer_values=endpoint.Bank.layer_values


def configs():
    result=[]
    for cfg in original.configs():
        # Retain both bare contractions, and rational / interval-only C5
        # controls. Test the endpoint composition under both contractions.
        if not cfg.get('factorized_full') or cfg['contract_rounds']==5:result.append(cfg)
        if cfg.get('interval_verifier'):
            result.append(dict(cfg,name=cfg['name']+'_endpoints',endpoint_bound=True,interval_comparator=cfg['name']))
    return result


def fit(x,v,state,cfg,rng,count=2048):
    if not cfg.get('endpoint_bound') or state is not None:return original.fit(x,v,state,cfg,rng,count)
    previous=original.IntervalBank
    try:
        original.IntervalBank=EndpointIntervalBank
        prediction,new,meta=original.fit(x,v,state,cfg,rng,count)
    finally:original.IntervalBank=previous
    meta['bound_kind']='duplicate_free_tent_endpoints'
    return prediction,new,meta


def verify():
    rng=np.random.default_rng(5900001);b=rng.uniform(-.12,.12,4);xx=rng.uniform(0,1,24)
    vv=base.forward(xx,b)+np.random.default_rng(24900001).uniform(-.001,.001,24);x=xx[:4];v=vv[:4]
    count=0;proofs=0;local_cfg=None
    for cfg in [c for c in configs() if c.get('endpoint_bound')]:
        refcfg=dict(cfg,name=cfg['interval_comparator'],endpoint_bound=False)
        ref,reference,rm=original.fit(x,v,None,refcfg,np.random.default_rng(482501),64)
        pred,state,meta=fit(x,v,None,cfg,np.random.default_rng(482501),64)
        assert reference.samples.tobytes()==state.samples.tobytes() and reference.anchor.tobytes()==state.anchor.tobytes()
        assert ref(xx).tobytes()==pred(xx).tobytes()
        for key in ['credit_proofs','credit_bank','positive_mode_keys','discovery_bank_sha256','completion_proposals','geometry_calls']:
            assert meta[key]==rm[key],key
        assert [c['input_pattern_hash'] for c in meta['screen_calls']]==[c['input_pattern_hash'] for c in rm['screen_calls']]
        assert [c['screened_pattern_hash'] for c in meta['screen_calls']]==[c['screened_pattern_hash'] for c in rm['screen_calls']]
        count+=1;proofs+=len(meta['credit_proofs'])
        if cfg['name']=='alm_native_full_c5_interval_endpoints':local_cfg=cfg
    core=neighbor.previous.core;jac=base.forward_jacobian;refine=core.batched.refine
    def forbidden(*a,**k):raise AssertionError('global BP in local endpoint-interval H2')
    try:
        base.forward_jacobian=forbidden;core.batched.refine=forbidden;state=None
        for n in [4,8,16,24]:_,state,_=fit(xx[:n],vv[:n],state,local_cfg,np.random.default_rng(482501+n),64)
    finally:base.forward_jacobian=jac;core.batched.refine=refine
    return dict(passed=True,endpoint_configs_bytewise=count,unchanged_proof_records=proofs,
        local_four_stage_no_global_bp=True,scope='old seed only; endpoint-interval composition, not timing')


if __name__=='__main__':
    import json
    print(json.dumps(verify(),indent=2),flush=True)
