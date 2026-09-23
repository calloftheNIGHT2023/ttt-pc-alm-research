"""252: atomic adjacent-dual intervention and support-only causal controls."""
import numpy as np
import cold_stagnation_switch as cold
import local_dual_jump as jump
base=cold.base
METHODS=['unchanged','bias_only','reorder_no_dual','activity_only','dual_jump','alm1','nodual1','pc1','branch_probe']

def retain(bank,incumbent,x,v):
    best=incumbent.copy();old_error,old_move=base.score(best[None],x,v,np.zeros(4))
    for b in bank:
        error,move=base.score(b[None],x,v,np.zeros(4))
        if base.better(error,move,old_error,old_move)[0]:best=b.copy();old_error,old_move=error,move
    return best

def bias_pass(b,h,u,x):
    ans=b.copy();r=len(b)
    for j in range(4):
        prev=np.broadcast_to(x,(r,len(x))) if j==0 else h[j-1]
        ans[:,j]=base.bias_solve(prev,h[j]+u[j],b[:,j].copy(),0.,float('inf'),.01)
    return ans

def run(b,h,incumbent,x,v,event,method):
    current=b.copy();activity=h.copy();dual=np.zeros_like(h);extra={};trial_roles=[]
    if method in ['alm1','nodual1','pc1']:
        state=cold.Local(b[None],x,v,method[:-1]);state.h=h[:,None].copy();state.best=retain(b[None],incumbent,x,v)[None]
        state.errors,state.moves=base.score(state.best,x,v,np.zeros(4));state.step()
        current=state.b[0].copy();activity=state.h[:,0].copy();dual=state.u[:,0].copy()
        extra=dict(activity_blocks=16,bias_blocks=4,dual_updates=16 if method=='alm1' else 0)
    elif method=='branch_probe':
        activities=[h.copy()];trial_roles=[dict(kind='source')]
        for j in range(3):
            for i in range(len(x)):
                for row in jump.values(jump.scalars(x,b,h,j,i),0.)['branches']:
                    ah=h.copy();ah[j,i]=row['z'];activities.append(ah);trial_roles.append(dict(j=j,i=i,k=row['k']))
        all_h=np.stack(activities,axis=1);all_u=np.zeros_like(all_h);all_b=np.tile(b,(len(activities),1))
        all_b[1:]=bias_pass(all_b[1:],all_h[:,1:],all_u[:,1:],x)
        index,_=cold.select(all_b,x,v);current=all_b[index].copy();activity=all_h[:,index].copy();dual=all_u[:,index].copy()
        extra=dict(activity_branch_proposals=len(activities)-1,bias_blocks=4*(len(activities)-1),selected_trial=index)
    elif method!='unchanged':
        if event is not None and method in ['reorder_no_dual','activity_only','dual_jump']:
            j,i=event['j'],event['i']
            if method=='reorder_no_dual':
                rows=jump.values(jump.scalars(x,b,h,j,i),0.)['branches'];activity[j,i]=min(rows,key=lambda r:(r['energy'],r['k']))['z']
            else:
                activity[j,i]=event['z']
                if method=='dual_jump':dual[j,i],dual[j+1,i]=event['u']
        current=bias_pass(b[None],activity[:,None],dual[:,None],x)[0]
        extra=dict(activity_blocks=int(event is not None and method!='bias_only'),bias_blocks=4,dual_writes=2 if event is not None and method=='dual_jump' else 0)
    if method!='branch_probe':all_b=current[None];all_h=activity[:,None];all_u=dual[:,None];trial_roles=[dict(kind='result')]
    best=retain(np.r_[b[None],all_b],incumbent,x,v)
    arrays=dict(b=current,h=activity,u=dual,best=best,trial_b=all_b,trial_h=all_h,trial_u=all_u)
    return arrays,dict(**extra,trial_roles=trial_roles)
