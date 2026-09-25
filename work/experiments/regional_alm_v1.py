"""376 exact local augmented blocks on a tested region; strict proofs alone reject."""
import time
import numpy as np
import local_region_screen as screen
from test_region_conditioned_credit_v1 import guarded

TAU=.01
METHODS=['regional_active128','regional_passive128','regional_instant128','regional_active1024','regional_passive1024',
         'pdhg_cold128','pdhg_cold1024','pdhg_box128','pdhg_box1024']
PRIMARY='regional_active128'


def previous(x,h):return np.concatenate([np.broadcast_to(x,(len(h),1,len(x))),h[:,:-1]],axis=1)
def residual(x,s,c,b,z,h):return z-previous(x,h)-b[:,:,None],h-s*z-c
def energy(x,s,c,b,z,h,p,a):
    rp,ra=residual(x,s,c,b,z,h);return .5*((rp+p)**2+(ra+a)**2).sum((1,2))


def block_b(x,b,z,h,p):return np.clip(((z-previous(x,h)+p).sum(2)+TAU*b)/(len(x)+TAU),-.12,.12)
def block_z(x,b,z,h,p,a,s,c,zl,zh):return np.clip((previous(x,h)+b[:,:,None]-p+s*(h-c+a)+TAU*z)/(1+s*s+TAU),zl,zh)
def block_h(b,z,h,p,a,s,c,hl,hh):
    num=s*z+c-a+TAU*h;den=np.full((1,h.shape[1],1),1+TAU)
    num[:,:-1]+=z[:,1:]-b[:,1:,None]+p[:,1:];den[:,:-1]+=1
    return np.clip(num/den,hl,hh)


def initialize(x,v,regs,project):
    r,d,n=regs.shape;b=np.zeros((r,d));z=np.empty((r,d,n));h=np.empty_like(z);prev=x
    for j in range(d):z[:,j]=prev+b[:,j,None];h[:,j]=screen.base.g(z[:,j]);prev=h[:,j]
    box=screen.boxes(v,regs)
    if project:z=np.clip(z,box[0],box[1]);h=np.clip(h,box[2],box[3])
    return b,z,h,box


def solve(x,v,regs,name):
    start=time.perf_counter();assert name in METHODS;r,d,n=regs.shape;steps=1024 if name.endswith('1024') else 128
    pdhg=name.startswith('pdhg');project='cold' not in name;active=name.startswith('regional_active');instant=name.startswith('regional_instant')
    b,z,h,box=initialize(x,v,regs,project);zl,zh,hl,hh=box;s=screen.base.SLOPES[regs];c=screen.base.INTERCEPTS[regs]
    assert np.all(zl<=zh) and np.all(hl<=hh)
    p=np.zeros_like(z);a=p.copy();lastp=p.copy();lasta=a.copy();initial={'initial_b':b.copy(),'initial_z':z.copy(),'initial_h':h.copy()}
    zero=np.zeros_like(p);bb=b.copy();zb=z.copy();hb=h.copy()
    tb=.99/n;tz=.99/(1+abs(s));th=np.full((1,d,1),.99/2);th[:,-1]=.99;sp=np.full((1,d,1),.99/3);sp[:,0]=.99/2;sa=.99/(1+abs(s))
    first=np.full(r,-1,np.int32);proofp=np.zeros_like(p);proofa=np.zeros_like(a);lower=np.zeros(r);kind=np.full(r,-1,np.int8)
    checkpoints=[];proposal_rows=0;certified_checks=0
    def check(step,rp,ra):
        nonlocal lastp,lasta,proposal_rows,certified_checks
        for code,(pp,aa) in enumerate([(p,a),(p-lastp,a-lasta),(rp,ra)]):
            proposal_rows+=r;rough=screen.float_bound(x,v,regs,pp,aa)
            ids=np.flatnonzero((rough>1e-10*(1+abs(pp).sum((1,2))+abs(aa).sum((1,2))))&(first<0))
            if len(ids):
                vals=screen.certified_lower_bound(x,v,regs[ids],pp[ids],aa[ids]);certified_checks+=len(ids);good=vals>0;accepted=ids[good]
                first[accepted]=step;proofp[accepted]=pp[accepted];proofa[accepted]=aa[accepted];lower[accepted]=vals[good];kind[accepted]=code
        lastp=p.copy();lasta=a.copy();checkpoints.append(dict(step=step,certified=int((first>=0).sum()),seconds=time.perf_counter()-start,proposal_rows=proposal_rows,rounded_checks=certified_checks))
    check(0,*residual(x,s,c,b,z,h))
    with guarded():
        for it in range(steps):
            if pdhg:
                p+=sp*(zb-previous(x,hb)-bb[:,:,None]);a+=sa*(hb-s*zb-c);oldb,oldz,oldh=b,z,h
                b=np.clip(oldb+tb*p.sum(2),-.12,.12);z=np.clip(oldz-tz*(p-s*a),zl,zh)
                hc=a.copy();hc[:,:-1]-=p[:,1:];h=np.clip(oldh-th*hc,hl,hh)
                bb,zb,hb=2*b-oldb,2*z-oldz,2*h-oldh
            else:
                pp,aa=(p,a) if active else (zero,zero)
                b=block_b(x,b,z,h,pp);z=block_z(x,b,z,h,pp,aa,s,c,zl,zh);h=block_h(b,z,h,pp,aa,s,c,hl,hh)
            rp,ra=residual(x,s,c,b,z,h)
            if not pdhg and not instant:p+=.5*rp;a+=.5*ra
            if (it+1)%32==0 or it+1==steps:check(it+1,rp,ra)
    arrays={**initial,'final_b':b,'final_z':z,'final_h':h,'final_p':p,'final_a':a,'first_step':first,'proof_p':proofp,'proof_a':proofa,'proof_lower':lower,'proof_kind':kind}
    named=[b,z,h,p,a,lastp,lasta,bb,zb,hb,zero,zl,zh,hl,hh,s,c,tz,sa,rp,ra,first,proofp,proofa,lower,kind]
    meta=dict(method=name,steps=steps,active_feedback=active,passive_accumulation='passive' in name,project_initial=project,
        checkpoints=checkpoints,proposal_rows=proposal_rows,rounded_checks=certified_checks,total_seconds=time.perf_counter()-start,
        live_named_array_bytes=sum(t.nbytes for t in named),initial_archive_bytes=sum(t.nbytes for t in initial.values()),
        returned_array_bytes=sum(t.nbytes for t in arrays.values()),memory_scope='Named arrays only, not process peak',
        query_targets_accessed=False,suffix_accessed=False,global_bp_used=False,lp_used=False)
    return arrays,meta
