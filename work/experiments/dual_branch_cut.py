"""Exact local Lagrangian cuts over branch labels; no global derivative."""
from __future__ import annotations
from fractions import Fraction as F
import numpy as np
import streaming_branch_projection as base


class BranchCut:
    def __init__(self,regs,cert):
        self.shape=regs.shape; self.old=regs.ravel().copy()
        self.d0=F(int(cert["numerator"]),int(cert["denominator"]))
        assert self.d0>0
        bound=F(base.BOUND)
        lo=[-bound,F(0),F(1,2),F(1)]; hi=[F(0),F(1,2),F(1),1+bound]
        slopes=[0,2,-2,0]; offsets=[0,0,2,0]
        p=np.asarray(cert["up"]).ravel(); a=np.asarray(cert["ua"]).ravel()
        self.cost=[]
        for pf,af in zip(p,a):
            pp,aa=F(float(pf)),F(float(af)); row=[]
            for j in range(4):
                coef=pp-slopes[j]*aa
                row.append(coef*(lo[j] if coef>=0 else hi[j])-aa*offsets[j])
            self.cost.append(row)
        self.offset=self.d0-sum((row[int(j)] for row,j in zip(self.cost,self.old)),F(0))
        self.floatcost=np.array([[float(c) for c in row] for row in self.cost])
        self.floatoffset=float(self.offset)

    def exact_bound(self,regs):
        return self.offset+sum((row[int(j)] for row,j in zip(self.cost,regs.ravel())),F(0))

    def excludes(self,regs):
        screen=self.floatoffset+self.floatcost[np.arange(len(self.old)),regs.ravel()].sum()
        if screen<=1e-12: return False
        return self.exact_bound(regs)>0

    def proposals(self):
        choices=[min(range(4),key=lambda j:row[j]) for row in self.cost]
        gains=[row[int(old)]-row[alt] for row,old,alt in zip(self.cost,self.old,choices)]
        order=sorted(range(len(gains)),key=lambda k:gains[k],reverse=True)
        total=F(0); min_changes=None
        for m,k in enumerate(order,1):
            total+=gains[k]
            if total>=self.d0: min_changes=m; break
        if min_changes is None:
            return [],{"minimum_branch_changes":None,"whole_union_infeasible":True}
        proposed=[]
        for count in [min_changes,min_changes+1,min_changes+2]:
            out=self.old.copy()
            for k in order[:count]: out[k]=choices[k]
            out=out.reshape(self.shape)
            assert self.exact_bound(out)<=0
            if not any(np.array_equal(out,r) for r in proposed): proposed.append(out)
        return proposed,{"minimum_branch_changes":min_changes,"whole_union_infeasible":False,
                         "positive_gain_coordinates":sum(g>0 for g in gains)}
