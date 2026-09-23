"""287 fixed27 catalogue: frozen19 resource competitors plus eight original heads."""
import json
import probe_credit_resource_suite as resources
PRIMARY=resources.PRIMARY


def catalogue(root):
    p=json.loads((root/'results/probe_credit_confirmation/planning/protocol.json').read_text())
    old={c['name']:c for c in resources.legacy.original.configs()}
    additional=[dict(name=name,family='legacy',group='additional_head',config=old[name]) for name in p['additional_regression_head_names']]
    ans=p['core_configurations']+additional
    assert len(ans)==len({c['name'] for c in ans})==27 and len(additional)==8
    return ans


def fit(cfg,x,v,q,seed,loaded):
    return resources.fit(cfg,x,v,q,seed,loaded,trace=False)
