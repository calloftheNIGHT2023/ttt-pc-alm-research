"""271: one action per restart, 17 total continuations, no reference-file access.

Adapters are scoped to one serial call and restored in finally. The live engine
still performs preparation, certificate search, geometry and readout itself.
"""
from collections import Counter
import numpy as np
import live_minimum_dual as live

PRIMARY='mixed_DA_even64'
CONFIGS=[dict(name=PRIMARY,even='D',odd='A'),dict(name='mixed_DA_odd64',even='A',odd='D'),
         dict(name='mixed_RA_even64',even='R',odd='A'),dict(name='mixed_BA_even64',even='B',odd='A')]


def fit(cfg,x,v,q,seed,trace=False):
    actions=[cfg['even'] if r%2==0 else cfg['odd'] for r in range(17)]
    assert set(actions)<=set('DARB')
    original_transfer=live.transfer.run
    original_local=live.cold.Local
    count=0;continuations=0;history={};effective_initial=None

    def transfer(b,h,incumbent,x,v,event,method):
        nonlocal count
        assert method=='dual_jump' and count<17
        action=actions[count];count+=1
        # Exactly one original action, never both actions followed by selection.
        arrays,metadata=original_transfer(b,h,incumbent,x,v,event,{'D':'dual_jump','R':'dual_jump','A':'activity_only','B':'bias_only'}[action])
        return arrays,metadata

    class ScopedLocal(original_local):
        def step(self):
            nonlocal continuations,effective_initial
            if self.method=='alm':
                assert self.b.shape==(17,4)
                if self.step_count==0:
                    self.u[:,np.array(actions)=='R']=0.
                    effective_initial=self.u.copy()
                    if trace:
                        for k,value in self.arrays().items():history[k]=[value]
                super().step();continuations+=1
                if trace:
                    for k,value in self.arrays().items():history[k].append(value)
            else:
                assert self.method=='nodual'
                super().step()

    try:
        live.transfer.run=transfer;live.cold.Local=ScopedLocal
        arrays,metadata=live.fit(dict(initial='dual_jump',method='alm',steps=64),x,v,q,seed)
    finally:
        live.transfer.run=original_transfer;live.cold.Local=original_local
    assert count==17 and continuations==64 and effective_initial is not None
    arrays['assigned_actions']=np.array(actions)
    arrays['effective_initial_u']=effective_initial
    if trace:
        arrays.update({'history_'+k:np.array(values) for k,values in history.items()})
    metadata.update(assigned_actions=actions,action_counts=dict(Counter(actions)),atomic_calls=count,
        continuation_sweeps=continuations,total_restart_sweeps=17*continuations,
        trace_enabled=trace,scope='one selected atomic action per restart; no extra trajectories or reference reads')
    return arrays,metadata
