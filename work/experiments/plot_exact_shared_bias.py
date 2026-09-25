"""Layout-only figure regeneration, preserving audited solver sources."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import exact_shared_bias_block as block
p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True);args=p.parse_args()
w=json.loads((args.results/'audit.json').read_text())['strict_coordinate_stall']
c=np.array([w['c']]);t=np.array([w['target']]);old=np.array([w['old_z']]);grid=np.linspace(-.3,.3,1001)
values=[block.energy(b,block.prox(c+b,t,old,w['trust']),c,t,old,w['trust']) for b in grid]
fig,ax=plt.subplots(figsize=(8,4.5),layout='constrained');ax.plot(grid,values,label='After exact elimination of z')
ax.scatter([0,w['new_bias']],[w['old_energy'],w['new_energy']],c=['#bc5a65','#007d8a'],zorder=5)
ax.annotate('Coordinate fixed point',(0,w['old_energy']),xytext=(-.26,.018),arrowprops={'arrowstyle':'->'})
ax.annotate('Joint global minimum',(w['new_bias'],w['new_energy']),xytext=(.025,.011),arrowprops={'arrowstyle':'->'})
ax.set(xlabel='Shared bias b',ylabel='Conditional local energy',title='Joint update escapes a coordinate fixed point',ylim=(-.0008,.027))
ax.legend(loc='upper left');ax.grid(alpha=.12);fig.savefig(args.results/'shared_bias_witness.png',dpi=160);plt.close(fig)
