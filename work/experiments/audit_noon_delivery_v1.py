"""Check overview tables and links against current sealed scientific artifacts."""
from fractions import Fraction
from pathlib import Path
import re
from report_search_radius_development_v1 import read,sha,save


def main():
    root=Path(__file__).resolve().parents[2]
    file=root/'outputs/ttt-pc-alm-research/367_noon_research_delivery_20260925.md'
    text=file.read_text(encoding='utf-8');checks=0
    old={m['method']:m for m in read(root/'results/online_credit_fresh_pilot/pilot_evaluation_v2/methods.json')}
    oldmap={'预定候选 first-fit / dual':'online_first_fit_dual','先验特征岭回归16384':'cold__prior16384_ridge',
        '元训练岭回归128':'cold__meta_ridge128','元训练浅层头64/20':'cold__meta_shallow64_20',
        '同前缀Adam3840':'probe_then_adam3840_33','同触发随机符号信用':'online_first_fit_random_sign','延长ALM母求解':'probe_all_alm64'}
    for label,name in oldmap.items():
        m=old[name];assert f"| {label} | {m['metrics']['mse257']:.10f} | {m['mean_seconds']:.6f} |" in text;checks+=2
    new={m['method']:m for m in read(root/'results/direct_language/development_evaluation_v1/methods.json')}
    newmap={'必要语言直接全池，无母轨迹':'direct_language_all','当前研究主配置：ALM信用前缀':'frontier_dual_frontier_g8',
        'ALM母轨迹＋C20完整池':'frontier_c20_all','Adam240＋共同读出':'frontier_native_adam240'}
    for label,name in newmap.items():
        m=new[name];assert f"| {label} | {m['metrics']['mse257']:.10f} | {m['mean_current_seconds']:.6f} |" in text;checks+=2
    witness=read(root/'results/multiplier_fixed_point/witness/query_risk.json')
    assert witness['passed'] and f"{witness['uniform_risk']:.10f}" in text and f"{witness['initial_uniform_risk']:.10f}" in text
    assert f"{100*witness['relative_risk_reduction']:.2f}%" in text;checks+=3
    mech=read(root/'results/direct_language/development_evaluation_v1/mechanisms.json')
    assert sum(m['enumerated'] for m in mech)==277604 and sum(m['remaining'] for m in mech)==2409
    assert sum(m['direct_positive'] for m in mech)==sum(m['mother_all_positive'] for m in mech)==631
    assert all(m['bitwise_same_prediction'] and not m['direct_only'] and not m['mother_only'] for m in mech)
    assert sum(m['classifications'].get('zero_volume_or_empty',0) for m in mech)==6;checks+=5
    a=Fraction(3,25);d=a/2;assert 4*a*a/Fraction(3)*(d/a)**3*(1-d/a)**3==Fraction(3,10000);checks+=1
    links=[]
    for relative in re.findall(r'\]\(([^)]+)\)',text):
        path=(file.parent/relative).resolve();assert path.is_file() and path.is_relative_to(root)
        links.append(dict(file=str(path.relative_to(root)).replace('\\','/'),sha256=sha(path)))
    output=root/'results/direct_language/report_v1/qa_noon_overview.json'
    save(output,dict(passed=True,numerical_checks=checks,local_links=len(links),links=links,
        overview_sha256=sha(file),source_sha256=sha(Path(__file__)),
        note='Linked evidence hashes checked locally; historical claims remain historical, not rerun full experiments.'))
    print(dict(passed=True,numerical_checks=checks,local_links=len(links)),flush=True)


if __name__=='__main__':main()
