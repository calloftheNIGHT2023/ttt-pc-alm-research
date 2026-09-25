import argparse,json
from fractions import Fraction
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import vector_interval_memory as m
p=argparse.ArgumentParser();p.add_argument('--certificate',type=Path,required=True);args=p.parse_args()
data=json.loads(args.certificate.read_text());weights=m.family.make_weights(3,8)
fig,axes=plt.subplots(2,2,figsize=(10,8),layout='constrained')
for index,record in enumerate(data['records']):
    rng=np.random.default_rng(record['seed']);rng.uniform(-m.PRIOR,m.PRIOR,(3,8));x=rng.uniform(-1,1,(24,8));q=rng.uniform(-1,1,(512,8))
    bank=np.array([[[float(Fraction(v)) for v in row] for row in record[name]] for name in ['lower_task','upper_task']])
    for col,inputs in enumerate([x[:8],q]):
        pred=m.forward(bank,inputs,weights);ax=axes[index,col];ax.plot([-1,1],[-1,1],color='#8e969c',lw=1)
        ax.scatter(pred[0].ravel(),pred[1].ravel(),s=14 if col==0 else 5,alpha=.7 if col==0 else .18,color='#007d8a' if col==0 else '#b26780')
        ax.set(xlim=(-1.05,1.05),ylim=(-1.05,1.05),xlabel='Task A output',ylabel='Task B output',aspect='equal',
            title=f'Seed {record["seed"]}: '+('8 observed vectors' if col==0 else '512 unseen vectors'))
        ax.grid(alpha=.1)
        if col:ax.text(.03,.97,f'Query separation MSE: {record["query_separation_numerical"]:.5f}',transform=ax.transAxes,va='top',fontsize=9)
fig.suptitle('Same observed function values; different possible query functions',fontsize=14)
fig.savefig(args.certificate.with_name('null_fiber_witness.png'),dpi=160);plt.close(fig)
