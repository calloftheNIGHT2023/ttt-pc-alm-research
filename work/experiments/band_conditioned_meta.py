"""Frozen round81 learned regression and shallow controls; no new training."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
import matched_shifted_meta_models as models

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def load(root):
    folder=Path(root)/'results/scalar_matched_batched/meta_prior';p=json.loads((folder/'protocol.json').read_text());training=json.loads((folder/'training_summary.json').read_text())
    for name,h in p['source_sha256'].items():assert sha(Path(__file__).with_name(name))==h
    loaded={};manifest={}
    for cfg,t in zip(p['configs'],training):
        assert cfg['name']==t['method'];path=folder/(cfg['name']+'.pt');assert sha(path)==t['checkpoint_sha256']
        model=models.make_model(cfg);model.load_state_dict(torch.load(path,map_location='cpu',weights_only=True));model.eval()
        loaded[cfg['name']]=model
        manifest[cfg['name']]=dict(checkpoint_sha256=sha(path),shared_model_bytes=sum(a.numel()*a.element_size() for a in model.state_dict().values()),historical_training=t)
    return loaded,manifest

def arrays(state):
    return {f'fast_{i}':p.detach().numpy().copy() for i,p in enumerate(state if isinstance(state,tuple) else [state])}

def fit(model,x,v,state):
    # Input conversion and closed solve / gradient updates are inside timing.
    with torch.no_grad():new=model.adapt(torch.as_tensor(x[None]),torch.as_tensor(v[None]),state)
    def predict(q):
        with torch.no_grad():return model.predict(new,torch.as_tensor(q[None])).clamp(0,1)[0].numpy()
    meta=dict(persistent_state_bytes=sum(p.numel()*p.element_size() for p in (new if isinstance(new,tuple) else [new])),
        shared_model_bytes=sum(p.numel()*p.element_size() for p in model.state_dict().values()),clipping='Same original frozen meta evaluator: [0,1]')
    return predict,new,meta
