"""258A: short-circuit the identical lexicographic accepted-event selection.

Acceptors and binary writes are the frozen251 functions. No model state is
changed by a trial; skipping later trials cannot alter the first accepted one.
"""
from fractions import Fraction as F
import local_dual_jump as original

def scan(x,b,h):
    work=[];blocks=[];counts=dict(float_zero=0,float_tau=0,exact_zero=0,exact_tau=0,rounded_write=0,zero_pruned=0)
    for j in range(len(b)-1):
        for i in range(len(x)):
            fs=original.scalars(x,b,h,j,i);zero=original.values(fs,0.);counts['float_zero']+=1;k0=zero['current_branch']
            own=next((r for r in zero['branches'] if r['k']==k0),None);others=[r for r in zero['branches'] if r['k']!=k0]
            eligible=own is not None and bool(others) and all(own['energy']<r['energy'] for r in others)
            blocks.append(dict(j=j,i=i,zero_eligible=eligible,trials=[]));work.append(dict(j=j,i=i,fs=fs,zero=zero,rs=None,exact_zero=None))
    for tau in original.TAUS:
        for row,block in zip(work,blocks):
            trial=dict(tau=tau,accepted=False);block['trials'].append(trial)
            if not block['zero_eligible']:counts['zero_pruned']+=1;trial['reason']='zero condition fails';continue
            fv=original.values(row['fs'],tau);fd=original.classify(row['zero'],fv);counts['float_tau']+=1;trial['float_positive']=bool(fd['accepted'])
            if not fd['accepted']:continue
            if row['rs'] is None:
                row['rs']=original.scalars(x,b,h,row['j'],row['i'],True);row['exact_zero']=original.values(row['rs'],F(0),True);counts['exact_zero']+=1
            ev=original.values(row['rs'],F(tau),True);ed=original.classify(row['exact_zero'],ev);counts['exact_tau']+=1;trial['exact_positive']=bool(ed['accepted'])
            if not ed['accepted']:continue
            rounded=original.check_rounded(row['rs'],fv,fd);counts['rounded_write']+=1;trial['rounded_accepted']=bool(rounded['accepted'])
            if not rounded['accepted']:continue
            trial['accepted']=True;event=dict(tau=tau,j=row['j'],i=row['i'],k=fd['competing_branch'],z=float(fd['z']),u=list(map(float,fv['u'])),
                exact_margin=original.exact.pack(ed['margin']),rounded_margin=rounded['margin'])
            return dict(selected=event,candidates=[event],blocks=blocks,counts=counts)
    return dict(selected=None,candidates=[],blocks=blocks,counts=counts)
