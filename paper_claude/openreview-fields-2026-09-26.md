# ICLR 2027 OpenReview fields — revised after reports 448–450

## Title

When Does Local Credit Help Test-Time Learning? Noise-Calibrated TTT $\times$ PC-ALM Beyond Regression

## Authors

Keep the existing author list and order unless both authors explicitly decide to reorder it. Do not add or remove authors after the abstract deadline.

## Keywords

test-time learning, test-time training, local credit assignment, predictive coding, augmented Lagrangian methods, few-shot adaptation, nonlinear inverse problems, regression baselines

## TL;DR

Noise-calibrated TTT $\times$ PC-ALM uses posterior local inversion, relaxed multiplier credit, and training-only regime selection to outperform gradient TTT, closed-form nonlinear regression, and a Transformer on held-out multi-branch tasks built from synthetic data, language and graph attributes, and image phase retrieval, except for a mean tie at the near-identifiability vision limit.

## Abstract

Test-time learning updates internal state from context, but extra iterative state is unnecessary when closed-form regression on fixed features suffices and can become unstable when local constraints are incompatible with noisy observations. We study when layer-local augmented-Lagrangian credit is useful for multi-branch few-shot prediction. Our method inserts PC-ALM into a TTT layer: per-example local inversion proposes latent branches, accumulated multipliers coordinate them, and a local least-squares step updates memory. For noisy observations, we replace hard inversion with a posterior mean, relax multipliers according to estimated residual variance, correct that estimate for degrees of freedom, use a folded-normal readout, and mix solutions near the identifiability limit. We derive six requirements for the studied observation model---observation consistency, noise-matched dual relaxation, unbiased noise estimation, posterior mixing near non-identifiability, training-only regime selection, and decodable representations---and connect each violation to an observed failure mode. On an end-to-end synthetic in-context task, the learned TTT $\times$ PC-ALM layer recovers the hidden signal subspace (0.999 overlap) and succeeds on 90% of $n=4d$ episodes in two independent seeds, while official and stronger full-batch TTT-GD, closed-form ridge, and a longer-trained Transformer remain near normalized mean-squared error one. Across held-out language and graph attributes and CIFAR-10 phase-retrieval episodes, a member of the method family selected using training data reduces normalized mean-squared error relative to gradient TTT, quadratic closed-form regression, and the Transformer; paired evaluations reproduce the gains, except that at the $2m$ vision threshold it ties quadratic regression in mean. Specialized AMP-style solvers can match the method on some controlled inverse problems, and ordinary linearly separable few-shot classification remains a setting where closed-form heads are preferable.

## Primary Area

Keep: optimization

This remains the closest single area because the central contribution concerns augmented-Lagrangian test-time updates, local constrained optimization, fixed-point consistency, and noise-dependent solver behavior. Meta-learning/test-time adaptation should be emphasized through the title, keywords, and abstract.

## AI Assistance

Select all that apply below; do not select “No, not at all” or “Yes, but for none of the above purposes.”

- Yes, to aid or polish writing. Details are described in the paper.
- Yes, for retrieval and discovery (e.g., finding related work). Details are described in the paper.
- Yes, for research ideation or execution. Details are described in the paper.
- Yes, to draft sections of the paper. Details are described in the paper.
- Yes, for generating synthetic datasets. Details are described in the paper.
- Yes, for proving mathematical claims. Details are described in the paper.

The manuscript must contain a matching AI-use statement that describes the actual assistance and the authors' verification of AI-assisted code, proofs, experiments, interpretation, and prose.

## Fields that do not need a content change

- Code of Ethics: keep the acknowledgement if both authors have read and agree to it.
- Paper Visibility: keep the acknowledgement.
- Submission Requirements: keep the acknowledgement only after both authors have reviewed the requirements.
- License: CC BY 4.0 can remain.

## Evidence boundary used for this revision

- The end-to-end synthetic result is limited to the specified Hermite multi-branch task at $d=32$, with two seeds and fixed solver hyperparameter $\rho=1$.
- The real-domain results concern held-out attributes in NLP and graph representations and simulated phase-retrieval measurements of real CIFAR-10 images; they are not general NLP, graph-learning, or vision superiority claims.
- The method-family member is selected using training attributes or training images, not held-out test targets.
- The CV result at $2m$ is a mean tie with quadratic closed-form regression, despite a better per-sequence win rate.
- AMP-style inference is competitive on some controlled inverse problems, and closed-form heads remain preferable on ordinary linearly separable few-shot classification.
