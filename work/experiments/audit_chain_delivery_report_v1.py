"""Audit report table fields, display numbers, links and figure metadata."""
import argparse
import hashlib
import json
from pathlib import Path
import re
from PIL import Image


def read(p):
    return json.loads(p.read_text(encoding='utf-8'))


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def run(root, folder):
    m = read(folder/'manifest.json')
    for p, digest in m['inputs'].items():
        assert sha(root/p) == digest
    for name, digest in m['outputs'].items():
        assert sha(folder/name) == digest
    assert sha(root/'work/experiments/report_chain_delivery_v1.py') == m['source_sha256']
    report = (folder/'report.md').read_text(encoding='utf-8')
    base = root/'results/branch_image_chain'
    geo = read(base/'geometry_v1/summary.json')['aggregate']
    names = dict(dual='乘子信用', dual_plus_residual='乘子＋残差', residual='纯残差',
                 bp='BP 信用对照', random_sign='随机符号', zero='零信用')
    checked = 0
    for name, label in names.items():
        match = re.search(r'^\| '+re.escape(label)+r' \| (\d+) \| (\d+) \| (\d+) \|$', report, re.M)
        assert match
        assert list(map(int, match.groups())) == [geo[name][k] for k in ['proposal_pairs', 'positive_pairs', 'new_vs_prior']]
        checked += 3
    op = read(base/'opportunity_v1/summary.json')
    risk = op['risk_opportunity']['257']
    fraction = risk['no_trigger']['mean_contribution']/risk['all']['mean_contribution']
    expected = [100*fraction, 100*(1-fraction)]
    assert m['figure_values']['risk_percent'] == expected
    expected_mass = [100*op['aggregate']['dual'][k]['mean_mass']/op['mean_missing_mass']
                     for k in ['no_trigger', 'outside_window', 'ranked_out']]
    for a, b in zip(expected_mass, m['figure_values']['missing_mass_percent']):
        assert abs(a-b) < 1e-12
    assert f'{fraction*100:.2f}%' in report
    for r in risk.values():
        assert f"{r['mean_contribution']:.12g}" in report
    paths = re.findall(r'\]\(<([^>]+)>\)', report)
    assert paths and all(Path(p).exists() for p in paths)
    with Image.open(folder/'bottleneck.png') as im:
        assert im.size == (2000, 848)
    result = dict(passed=True, table_fields=checked, figure_fields=5,
                  local_links_checked=len(paths), image_pixels=[2000,848],
                  report_sha256=sha(folder/'report.md'), figure_sha256=sha(folder/'bottleneck.png'),
                  source_sha256=sha(Path(__file__)),
                  note='Visual inspection is separate; this checks numeric content and artifact integrity.')
    (folder/'qa_numeric.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(result)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--report', type=Path, required=True)
    args = p.parse_args()
    run(Path(__file__).resolve().parents[2], args.report)
