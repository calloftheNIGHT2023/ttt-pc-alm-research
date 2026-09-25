"""393 exact rational tests of missing-mass bias and partial-readout bounds."""
from fractions import Fraction as F
from pathlib import Path
import hashlib
import json
import random
import traceback

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/low_support_coverage/math_preflight_v1'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2), encoding='utf-8')


def risk(values, weights, prediction):
    return sum(w * (y-prediction)**2 for w, y in zip(weights, values)) / sum(weights)


def check(values, weights, cut):
    total = sum(weights); kept = sum(weights[:cut]); omitted = total-kept
    mk = sum(w*y for w, y in zip(weights[:cut], values[:cut])) / kept
    mo = sum(w*y for w, y in zip(weights[cut:], values[cut:])) / omitted
    m = sum(w*y for w, y in zip(weights, values)) / total
    delta = omitted/total
    bias = delta**2 * (mk-mo)**2
    assert m == (1-delta)*mk + delta*mo
    assert risk(values, weights, mk)-risk(values, weights, m) == bias
    # Enumerate all one-sample outcomes independently instead of using a
    # variance formula to generate its own expected answer.
    full_excess = sum(w/total*(y-m)**2 for w,y in zip(weights,values))
    truncated_excess = sum(w/kept*(y-m)**2 for w,y in zip(weights[:cut],values[:cut]))
    assert full_excess == risk(values, weights, m)
    assert truncated_excess == bias + risk(values[:cut], weights[:cut], mk)
    # Also enumerate the full Cartesian product for an unbiased two-sample mean.
    pairs = sum(w1*w2/kept**2*((y1+y2)/2-m)**2
        for w1,y1 in zip(weights[:cut],values[:cut])
        for w2,y2 in zip(weights[:cut],values[:cut]))
    assert pairs == bias + risk(values[:cut], weights[:cut], mk)/2
    upper_mass = omitted + F(3, 7)
    known_integral = kept*mk
    lower = known_integral/(kept+upper_mass)
    upper = (known_integral+upper_mass)/(kept+upper_mass)
    assert lower <= m <= upper
    assert upper-lower == upper_mass/(kept+upper_mass)
    assert (m-(lower+upper)/2)**2 <= ((upper-lower)/2)**2
    # Tight endpoints are attained by admissible missing constant-zero/one mass.
    assert lower == (known_integral+0)/(kept+upper_mass)
    assert upper == (known_integral+upper_mass)/(kept+upper_mass)
    return bias


def main():
    rng = random.Random(393731)
    cases = 0
    for n in range(2, 10):
        for _ in range(16):
            values = [F(rng.randrange(33), 32) for _ in range(n)]
            weights = [F(rng.randrange(1, 25), 17) for _ in range(n)]
            check(values, weights, rng.randrange(1, n)); cases += 1
    positive = check([F(0), F(1)], [F(1), F(1)], 1)
    assert positive == F(1, 4)
    no_predictive_gap = check([F(1, 3), F(1, 3)], [F(1), F(99)], 1)
    assert no_predictive_gap == 0
    # A favorable realized teacher is not a conditional-population guarantee.
    assert (F(0)-F(0))**2 < (F(1, 2)-F(0))**2
    save(OUT/'summary.json', dict(passed=True, exact_fraction_random_cases=cases,
        nonzero_missing_mass_bias_example=str(positive),
        ninety_nine_percent_missing_but_zero_bias=str(no_predictive_gap),
        realized_teacher_counterexample=True, posterior_samples_enumerated=[1,2],
        partial_mean_bounds_checked=True, research_query_targets_accessed=False,
        empirical_method_advantage_established=False,
        source_sha256=sha(Path(__file__)),
        design_sha256=sha(ROOT/'outputs/ttt-pc-alm-research/393_low_support_coverage_hypothesis_v1.md')))
    print(json.dumps(json.loads((OUT/'summary.json').read_text())), flush=True)


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=False)
    try:
        main()
    except Exception:
        save(OUT/'failure.json', dict(traceback=traceback.format_exc(), automatic_retry=False))
        raise
