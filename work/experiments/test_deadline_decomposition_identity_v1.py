"""Fast stdlib algebra checks for the frozen 391 decomposition, no task data."""
from pathlib import Path
import hashlib
import json
import math
import random
import decompose_deadline_risk_v1 as diagnostic


def mse(a, b):
    return math.fsum((u-v)**2 for u, v in zip(a, b)) / len(a)


def check(ra, rc, c, r, pa, pc, reference, target):
    actions_a = dict(constant=c, fallback=r, complete=pa)
    actions_c = dict(constant=c, fallback=r, complete=pc)
    direct = mse(actions_c[rc], target) - mse(actions_a[ra], target)
    jra, jrc = int(ra != 'constant'), int(rc != 'constant')
    jpa, jpc = int(ra == 'complete'), int(rc == 'complete')
    first = (jra-jrc) * (mse(c, target)-mse(r, target))
    second = (jpa-jpc) * (mse(r, target)-mse(reference, target))
    third = jpc*(mse(pc, target)-mse(reference, target))-jpa*(mse(pa, target)-mse(reference, target))
    error = abs(direct-math.fsum((first, second, third)))
    assert error < 1e-14
    return error, direct, second, third


def main():
    plan = diagnostic.read(diagnostic.PLAN/'protocol.json')
    for name, digest in plan['source_sha256'].items():
        assert diagnostic.sha(diagnostic.ROOT/name) == digest
    rng = random.Random(391171); maximum = 0.; cases = 0
    for shared in (False, True):
        for _ in range(32):
            c = [.5]*17
            r, pa, pc, reference, target = [[rng.random() for _ in range(17)] for _ in range(5)]
            if shared:
                pc = pa; reference = pa
            for ra in ('constant', 'fallback', 'complete'):
                for rc in ('constant', 'fallback', 'complete'):
                    error, _, _, correction = check(ra, rc, c, r, pa, pc, reference, target)
                    maximum = max(maximum, error); cases += 1
                    if shared:
                        assert correction == 0.
    _, beneficial, beneficial_part, _ = check('complete', 'fallback', [.5], [.5], [.1], [.1], [.1], [.1])
    _, harmful, harmful_part, _ = check('complete', 'fallback', [.5], [.1], [.9], [.9], [.9], [.1])
    assert beneficial > 0 and beneficial_part == beneficial
    assert harmful < 0 and harmful_part == harmful
    out = diagnostic.BASE/'mechanism_identity_preflight_v1'
    out.mkdir(parents=True, exist_ok=False)
    summary = dict(passed=True, algebra_cases=cases, maximum_error=maximum,
                   shared_prediction_correction_exactly_zero=True,
                   beneficial_and_harmful_completion_cases=2,
                   data_scope='independent random scalar/vector unit tests; no 387 support or query data',
                   query_targets_accessed=False,
                   frozen_plan_sha256=diagnostic.sha(diagnostic.PLAN/'protocol.json'),
                   source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    diagnostic.save(out/'summary.json', summary)
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
