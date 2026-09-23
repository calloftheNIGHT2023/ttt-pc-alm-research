"""Exact one-coordinate full-depth partition versus local tied partition."""
import argparse,hashlib,json
from fractions import Fraction as F
from pathlib import Path
import numpy as np
import light_tied_proposals as light


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--project',type=Path,required=True);args=parser.parse_args();rows=[]
    for depth in [2,4,6,8,10,12]:
        x=np.array([.5]);b=np.zeros(depth);h=light.forward_many(x,b[None])[1][0];u=np.zeros_like(h)
        exact=light.coordinate_intervals_exact(x,b,0)
        lower=(F(1,2)-F(light.B))*2**depth;upper=(F(1,2)+F(light.B))*2**depth
        expected=-(-upper.numerator//upper.denominator)-lower.numerator//lower.denominator
        _,_,meta=light.tied_free(x,b,h,u);assert len(exact)==expected and meta['segments']==2*(depth-1)
        rows.append(dict(depth=depth,full_forward_first_bias_intervals=len(exact),proved_interval_formula=expected,
                         all_hidden_tied_intervals=meta['segments'],tied_finite_candidates=meta['candidates']))
    result=dict(passed=True,source_sha256=sha(Path(__file__)),primitive_sha256=sha(Path(light.__file__)),rows=rows,
        theorem='for x=1/2, zero fixed biases and fixed B=0.12, the exact first-bias full-forward partition has ceil(2^L(x+B))-floor(2^L(x-B)) pieces, versus 2(L-1) total local tied pieces at this forward state',
        scope='partition work separation only; local candidates do not cover all full-depth pieces and have no universal risk dominance')
    out=args.project/'results/light_tied_routing/complexity.json';out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
