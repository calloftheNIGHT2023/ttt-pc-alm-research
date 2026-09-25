"""386--389 / 391--392 exact bounded publication; refuses unfinished evidence.

390 training prototypes are deliberately not included in this result release.
They need their own passed preflight and a separately reviewed publication.
"""
import argparse
import hashlib
from pathlib import Path
import re
import region_credit_snapshot_20260925 as pack

PARENT = '3560cbc2cdd91532f3fb7bf0ee4b8256b76b1058'
BASE = 'results/new_task_deadline_risk'
MANIFEST = 'release/deadline-risk-20260925.json'


def paths(root):
    tree = {}
    for record in pack.git(root, 'ls-tree', '-r', '-z', PARENT).decode().strip('\0').split('\0'):
        header, name = record.split('\t', 1); tree[name] = header.split()[2]
    files = set()

    def immutable(name, digest):
        name = name.replace('\\', '/')
        assert pack.sha(root/name) == digest, name
        if name in tree:
            payload = (root/name).read_bytes()
            assert tree[name] == hashlib.sha1(b'blob '+str(len(payload)).encode()+b'\0'+payload).hexdigest(), name
        else:
            files.add(name)

    def folder(relative, passed=True):
        location = root/relative
        if passed:
            summary = pack.read(location/'summary.json')
            assert summary['passed'] and not (location/'failure.json').exists(), relative
            for name, digest in summary.get('outputs_sha256', {}).items():
                assert pack.sha(location/name) == digest, (relative, name)
        files.update(p.relative_to(root).as_posix() for p in location.rglob('*') if p.is_file())

    protocol = pack.read(root/BASE/'development_v1/protocol.json')
    for name, digest in protocol['source_sha256'].items():
        immutable(name, digest)
    for name, digest in protocol['pretrained_sha256'].items():
        immutable(name, digest)
    for stage in ('preflight_v1', 'audit_preflight_v1', 'development_v1', 'audit_v1', 'evaluation_v1',
                  'mechanism_identity_preflight_v1', 'mechanism_decomposition_v1'):
        folder(BASE+'/'+stage)
    plan = pack.read(root/BASE/'mechanism_plan_v1/protocol.json')
    assert plan['frozen_before_query'] and plan['primary_protocol_unchanged']
    assert plan['task_protocol_sha256'] == pack.sha(root/BASE/'development_v1/protocol.json')
    for name, digest in plan['source_sha256'].items():
        immutable(name, digest)
    folder(BASE+'/mechanism_plan_v1', passed=False)
    for row in pack.read(root/BASE/'development_v1/rows.json'):
        for name, digest in row['files'].items():
            assert pack.sha(root/row['directory']/name) == digest
    for relative in ('results/n24_task_baselines/preflight_v2',
                     'results/n24_optimizer_controls/preflight_v1',
                     'results/deadline_prediction/preflight_v1'):
        folder(relative)
    failure = root/'results/n24_task_baselines/preflight_v1/failure.json'
    assert failure.exists() and pack.read(failure)['automatic_retry'] is False
    folder('results/n24_task_baselines/preflight_v1', passed=False)

    report = root/BASE/'report_v1'; manifest = pack.read(report/'manifest.json')
    qa = pack.read(report/'qa_numeric.json'); visual = pack.read(report/'visual_qa.json')
    assert qa['passed'] and qa['manifest_sha256'] == pack.sha(report/'manifest.json')
    assert visual['passed'] and visual['actually_viewed'] and visual['numeric_qa_sha256'] == pack.sha(report/'qa_numeric.json')
    for image in visual['images']:
        assert image['sha256'] == pack.sha(report/image['file'])
    for name, digest in manifest['outputs_sha256'].items():
        assert pack.sha(report/name) == digest
    assert manifest['source_sha256'] == pack.sha(root/'work/experiments/report_new_task_deadline_risk_v1.py')
    assert manifest['entry_sha256'] == pack.sha(root/manifest['entry_file'])
    folder(BASE+'/report_v1', passed=False)
    companion = root/BASE/'all_controls_report_v1'
    companion_manifest = pack.read(companion/'manifest.json')
    companion_qa = pack.read(companion/'qa_numeric.json')
    companion_visual = pack.read(companion/'visual_qa.json')
    assert companion_qa['passed'] and companion_qa['manifest_sha256'] == pack.sha(companion/'manifest.json')
    assert companion_visual['passed'] and companion_visual['actually_viewed']
    assert companion_visual['numeric_qa_sha256'] == pack.sha(companion/'qa_numeric.json')
    assert companion_visual['mechanism_summary_sha256'] == pack.sha(root/BASE/'mechanism_decomposition_v1/summary.json')
    for image in companion_visual['images']:
        assert image['sha256'] == pack.sha(companion/image['file'])
    for name, digest in companion_manifest['outputs_sha256'].items():
        assert pack.sha(companion/name) == digest
    assert companion_manifest['parent_numeric_qa_sha256'] == pack.sha(report/'qa_numeric.json')
    assert companion_manifest['evaluation_summary_sha256'] == pack.sha(root/BASE/'evaluation_v1/summary.json')
    assert companion_manifest['source_sha256'] == pack.sha(root/'work/experiments/report_all_deadline_controls_v1.py')
    assert companion_manifest['entry_sha256'] == pack.sha(root/companion_manifest['entry_file'])
    folder(BASE+'/all_controls_report_v1', passed=False)
    notes = ['outputs/ttt-pc-alm-research/386_deadline_risk_design_draft_v1.md',
             'outputs/ttt-pc-alm-research/389_cross_domain_small_model_requirements_v1.md',
             manifest['entry_file'], companion_manifest['entry_file'], 'release/DEADLINE-RISK-20260925.md']
    for doc in [report/'report.md', companion/'report.md', root/BASE/'mechanism_decomposition_v1/report.md',
                root/companion_manifest['entry_file'],
                root/manifest['entry_file'], root/'release/DEADLINE-RISK-20260925.md']:
        for link in re.findall(r'\]\(([^)]+)\)', doc.read_text(encoding='utf-8')):
            if not link.startswith(('https:', 'http:', '#')):
                assert (doc.parent/link.split('#')[0]).resolve().is_file(), (doc, link)
    files.update(notes)
    files.update(['scripts/deadline_risk_snapshot_20260925.py', 'scripts/region_credit_snapshot_20260925.py',
                  'work/experiments/report_new_task_deadline_risk_v1.py',
                  'work/experiments/report_all_deadline_controls_v1.py',
                  'work/experiments/test_deadline_decomposition_identity_v1.py'])
    return sorted(files)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('action', choices=('build', 'stage', 'verify')); args = parser.parse_args()
    pack.MANIFEST = MANIFEST; pack.paths = paths
    print(dict(action=args.action, **getattr(pack, args.action)(Path(__file__).resolve().parents[1])), flush=True)
