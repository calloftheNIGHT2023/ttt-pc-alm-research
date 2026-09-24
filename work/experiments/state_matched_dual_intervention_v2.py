"""290: restore the original .12 discovery context; preserve failed v1."""
import argparse
from pathlib import Path
import state_matched_dual_intervention_v1 as original
from posterior_confirmation_pipeline import discovery_box


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, required=True)
    root = parser.parse_args().project.resolve()
    out = root/'results/state_matched_dual_intervention/development_v2'
    out.mkdir(parents=True, exist_ok=False)
    raw = root/'results/certificate_activity_attribution/development'
    p = original.read(raw/'protocol.json')
    before = original.read(raw/'before_evaluation_manifest.json')
    assert original.sha(raw/'protocol.json') == before['protocol_sha256']
    assert p['discovery_bound'] == .12
    metadata = dict(discovery_bound=.12, default_module_bound=original.cold.base.BOUND,
        correction_note_sha256=original.sha(root/'outputs/ttt-pc-alm-research/290_bound_scope_correction.md'),
        failed_v1_sha256=original.sha(out.parent/'development_v1/failure.json'),
        source_sha256={n: original.sha(root/'work/experiments'/n) for n in
                      ['state_matched_dual_intervention_v1.py', 'state_matched_dual_intervention_v2.py']},
        original_protocol_sha256=original.sha(raw/'protocol.json'),
        original_manifest_sha256=original.sha(raw/'before_evaluation_manifest.json'),
        no_tolerance_relaxation=True, query_targets_accessed=False)
    original.save(out/'scope_correction.json', metadata)
    try:
        with discovery_box(p['discovery_bound']):
            assert original.cold.base.BOUND == .12
            original.run(root, out)
        for name, digest in metadata['source_sha256'].items():
            assert original.sha(root/'work/experiments'/name) == digest
        original.save(out/'wrapper_complete.json', dict(passed=True, discovery_bound=.12,
            summary_sha256=original.sha(out/'summary.json'),
            scope_correction_sha256=original.sha(out/'scope_correction.json'),
            core_research_goal_complete=False))
    except BaseException as exc:
        original.save(out/'failure.json', dict(error_type=type(exc).__name__, message=str(exc), no_automatic_retry=True))
        raise


if __name__ == '__main__':
    main()
