# Properties

The concept attribution of SPARC satisfies four properties that feature level attributions do not have, or do not have at the concept level. Each is checked numerically in `tests/test_properties.py` and `tests/test_concept_shapley.py`.

**Efficiency at the concept level.** For every membership matrix whose rows sum to one, the concept attributions of an instance sum to the model output minus the expected output over the background sample:

$$\sum_{k=1}^{K} \phi_k(x) = f(x) - \mathbb{E}[f], \qquad \phi_k(x) = \sum_{j=1}^{d} u_{jk}\,\varphi_j(x),$$

where \(\varphi_j\) is the feature level Shapley value and \(u_{jk}\) the membership of feature \(j\) in concept \(k\). With the reference centred point estimate `values.sum(axis=1) + base_values` equals `output` exactly.

**Reduction.** With the identity partition (one crisp concept per feature), one resample and one model class the method returns the feature level Shapley values exactly, and the fuzzy numbers degenerate to points.

**Symmetry.** Two concepts whose aggregated contributions coincide in every coalition receive the same attribution, inherited from the symmetry of the Shapley value applied to every pool member.

**Duplication invariance.** Adding an exact copy of a feature leaves the attribution of the concept that absorbs the copy unchanged, whereas the feature level attribution of the copied feature is split between the copies.

## The concept game

The direct estimator (`concept_estimator="direct"`) shows where these properties come from. Let every concept be a player of a cooperative game. A coalition \(S\) of concepts keeps feature \(j\) to the degree

$$w_j(S) = \min\Big(1, \sum_{k \in S} u_{jk}\Big),$$

the membership mass of \(j\) in \(S\); the grand coalition keeps every feature and the empty coalition removes every feature. The value of a coalition is the model output with the remaining features removed to the degree \(1 - w_j(S)\), by marginal imputation (the interventional expectation over the background) or by conditional imputation. The Shapley value of concept \(k\) in this game is the direct concept attribution. With marginal imputation it coincides with the membership aggregation of the interventional feature level values in practice, so the aggregation of Equation 1 is the Shapley value of a game over fuzzy concepts, not only a convenient sum. With crisp concepts the game is the grouped Shapley value, with the identity partition the feature level Shapley value; both reductions are tested exactly.

## Calibration and stability

Two further properties are empirical rather than axiomatic and are the subject of the study: the support and the core of the fuzzy numbers cover an independent replicate attribution at about their nominal levels (0.90 and 0.50), and the ranking of the concepts is stable across resamples where feature level rankings are not. The [regression tutorial](tutorials/regression.md) shows both on a problem with a known structure, and the study repository reports them on fifteen public benchmarks and four open access datasets.
