"""Show the proved point-versus-region distinction and oracle potential."""
import argparse,hashlib,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--project',type=Path,required=True);args=parser.parse_args();root=args.project/'results/tied_local_block/pool_audit'
    data=json.loads((root/'analysis.json').read_text());lookup={r['group']:r for r in data['summary']};count=data['independent_positive_regions']
    fig,(left,right)=plt.subplots(1,2,figsize=(16,7),gridspec_kw={'width_ratios':[1.25,1]},layout='constrained');left.set_axis_off();left.set_xlim(0,1);left.set_ylim(0,1)
    def box(x,y,text,color):left.text(x,y,text,ha='center',va='center',fontsize=14,bbox=dict(boxstyle='round,pad=.7',fc=color,ec='#505761'))
    def arrow(xy,start):left.annotate('',xy=xy,xytext=start,arrowprops=dict(arrowstyle='->',lw=2,color='#505761'))
    box(.49,.86,f'Tied proposal pool\n{count} previously missing valid modes','#dceef0')
    box(.23,.53,'Filter this point\nby its residual cuts','#f1dfd8');box(.77,.53,'Test the entire mode\nwith sound certificates','#dceef0')
    arrow((.23,.65),(.4,.78));arrow((.77,.65),(.6,.78))
    box(.23,.20,'0 new modes kept\nby point certification','#f1dfd8');box(.77,.20,f'{count} modes have exact\nfeasible parameter witnesses','#dceef0')
    arrow((.23,.31),(.23,.42));arrow((.77,.31),(.77,.42))
    left.set_title('An inadmissible point can belong to a useful mode',fontsize=16,pad=18)
    names=['Original archive','Add all scalar modes','Add all tied modes'];values=[lookup[g]['mean_ideal_truncation'] for g in ['before','scalar_all','tied_all']]
    bars=right.bar(names,values,color=['#8290a5','#bc705d','#167e89'],width=.6)
    for bar,value in zip(bars,values):right.text(bar.get_x()+bar.get_width()/2,value+max(values)*.025,f'{value:.3e}',ha='center',fontsize=12)
    right.set_ylim(0,max(values)*1.22);right.grid(axis='y',alpha=.2);right.set_axisbelow(True);right.tick_params(axis='x',labelrotation=10)
    right.set_ylabel('Mean estimated ideal truncation component');right.set_title('Potential if every discovered valid mode is added\nEvaluator-only analysis, not actual query MSE',fontsize=15)
    fig.suptitle('Candidate-pool audit: useful explanations exist before point filtering',fontsize=20)
    fig.text(.5,-.035,'No online gain established yet. The strong large-prior baseline already contains all five modes; routing and construction costs remain to be tested.',ha='center',fontsize=12)
    output=root/'tied_pool_routing.png';fig.savefig(output,dpi=160,bbox_inches='tight');plt.close(fig)
    audit=dict(source_sha256=sha(Path(__file__)),input_sha256=sha(root/'analysis.json'),figure_sha256=sha(output))
    (root/'figure_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8');print(json.dumps(audit,indent=2))


if __name__=='__main__':main()
