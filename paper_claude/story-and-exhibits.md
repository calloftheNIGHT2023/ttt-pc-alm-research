# Story and exhibit plan

## Working title

**When Does Local Credit Matter? Multiplier Dynamics versus Branch Search**

## Five-sentence story

1. Test-time learning can update a deep internal model from a few support observations, but a better query prediction may come from the credit signal, the branch-search procedure, the readout, or simply more computation.
2. Existing end metrics do not identify which component mattered, and a local-credit method can have a genuine dynamical effect without beating a matched global optimizer.
3. We make the distinction testable on a controlled compositional regression family by combining an exact block-stationary escape certificate with sealed unseen-query evaluation and same-enhancement controls.
4. Accumulated multipliers provably escape a class of exact shared-parameter traps, and the frozen candidate beats 21 of 25 preregistered controls over 8192 new tasks, but it does not separate from four same-start Adam continuations.
5. A later branch-frontier improvement is reproduced bitwise by residual-only credit, showing that multiplier mechanism, search coverage, and predictive advantage are three different claims.

## Exhibit plan

| Exhibit | Reviewer question | Evidence | Main takeaway | Placement |
|---|---|---|---|---|
| Fig. 1: mechanism witness | Can multipliers change the primal branch rather than merely add state? | Exact rational witness and query integration | Yes, conditionally: the multiplier changes relative branch energy and causes a finite-step escape. | Main text |
| Table 1: frozen confirmation | Does the candidate improve unseen-query error under a sealed protocol? | 8192 tasks, 27 methods, 221184 predictions | It improves over many controls but not matched same-start Adam. | Main text |
| Fig. 2: preregistered contrasts | Which comparisons pass the fixed criterion? | Independent risk and bootstrap audit | 21/25 pass; all four Adam contrasts remain unresolved. | Main text |
| Fig. 3: frontier risk and cost | Does better search coverage establish multiplier-specific benefit? | 32-task concurrent online evaluation | No: scheduling helps, but residual or zero credit reproduces or slightly improves the result. | Main text |
| Appendix Table A1 | Can every claim be traced to a decisive test? | Release manifests and audits | Separates proof, confirmation, development evidence, and negative controls. | Appendix opening |

## Draft order

1. Controlled task and information boundary.
2. Exact multiplier escape theorem and witness.
3. Frozen confirmation and same-start optimizer controls.
4. Branch-search frontier and attribution controls.
5. Introduction, related work, abstract, conclusion.
