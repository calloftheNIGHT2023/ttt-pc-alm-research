"""Check report tables, figure inputs, links and hashes independently."""
import argparse
from pathlib import Path
import re
from PIL import Image
from evaluate_complete_credit_mode_geometry_v1 import read,save,sha


def run(root,out):
    folder=root/'results/support_consistency_trigger/report_v1';base=folder.parent
    manifest=read(folder/'manifest.json');report=(folder/'report.md').read_text(encoding='utf-8')
    for n,d in manifest['inputs'].items():assert sha(root/n)==d
    for n,d in manifest['outputs'].items():assert sha(folder/n)==d
    assert sha(root/'work/experiments/report_support_consistency_v1.py')==manifest['source_sha256']
    risk={(r['method'],r['grid'],r['stratum']):r for r in read(base/'risk_v1/aggregate.json')}
    channels=['dual','dual_plus_residual','residual','bp','random_sign','zero'];labels=['乘子','乘子＋残差','残差','BP信用','随机符号','零信用']
    count=0
    for n,label in zip(channels,labels):
        a=risk['original+first_fit/'+n,257,'all']['conditional_excess'];b=risk['original+uniform_state/'+n,257,'all']['conditional_excess']
        assert f'| {label} | {a:.10f} | {b:.10f} |' in report;count+=2
    for n,r in read(base/'geometry_v1/summary.json')['aggregate'].items():
        assert f"| {n} | {r['positive_pairs']} | {r['new_vs_prior']} | {r['new_vs_prior_and_controls']} |" in report;count+=3
    for policy,key in [('first_fit','first_fit'),('uniform_state','uniform')]:
        assert manifest['figure_values'][key]==[risk['original+'+policy+'/'+n,257,'all']['conditional_excess']*1e3 for n in channels]
    er={(r['method'],r['grid']):r for r in read(base/'exclusive_value_v1/aggregate.json')}
    names=[p+'/'+n for p in ['first_fit','uniform_state'] for n in ['dual','dual_plus_residual']]
    assert manifest['figure_values']['marginal_gains']==[-er[n,257]['mean_delta']*1e5 for n in names]
    paths=re.findall(r'\]\(<([^>]+)>\)',report);assert all(Path(p).is_file() for p in paths)
    with Image.open(folder/'credit_case.png') as im:assert im.size==(2080,992)
    result=dict(passed=True,table_numeric_fields=count,figure_numeric_fields=16,local_links=len(paths),
                image_pixels=[2080,992],report_sha256=sha(folder/'report.md'),figure_sha256=sha(folder/'credit_case.png'),
                source_sha256=sha(Path(__file__)),visual_review_separate=True)
    save(out/'summary.json',result);print(result)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    run(Path(__file__).resolve().parents[2],a.out)
