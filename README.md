# SPARC

Shapley Pareto Robust Concept attribution with fuzzy logic: post hoc explanations of tabular black box models that are stable, calibrated and sparse.

Charis Ntakolia, Department of Aeronautical Studies, Hellenic Air Force Academy.

SPARC attributes a prediction to fuzzy concepts of correlated features instead of to single features, so that the credit is not split arbitrarily among features that carry the same information. Every attribution is estimated over bootstrap resamples and several model classes and reported as a trapezoidal fuzzy number with a sign confidence, a disagreement index between model classes and a linguistic label such as weakly positive. The number of concepts, the fuzzifier, the sparsity and the alpha level are selected from a Pareto front that trades off fidelity loss, complexity and instability; a knee point is reported by default and the whole front is kept.

## Installation

```
pip install sparc-xai
pip install "sparc-xai[plots]"     # with matplotlib
```

Python 3.10 or later. The package is imported as `sparc`.

## Quick start

```python
import numpy as np
from sklearn.model_selection import train_test_split

import sparc
from sparc.synthetic import make_synthetic

problem = make_synthetic("additive", n_samples=600, n_features=9, rho=0.8, n_factors=3, random_state=0)
X_train, X_test, y_train, y_test = train_test_split(problem.X, problem.y, random_state=0)

explainer = sparc.Explainer(n_resamples=20, point_estimate="reference", random_state=0)
explainer.fit(X_train, y_train, feature_names=[f"x{j}" for j in range(9)])
explanation = explainer.explain(X_test[:20])

print(explanation.describe(0))          # the linguistic explanation of the first instance
explanation.instance_frame(0)           # concepts, fuzzy attribution, sign confidence, labels
explanation.front                       # the Pareto front of explanation configurations
explanation.membership                  # feature by concept membership matrix
```

The default model classes are a gradient boosting model (the reference model), a random forest and a multilayer perceptron; pass `model_classes=[("gbm", GradientBoostingRegressor()), ...]` to change them. Classifiers are explained through the probability of `target_class`. `apriori_groups` adds a known grouping of the features as a candidate partition, `solver="nsga2"` searches large decision spaces, and `n_jobs` parallelizes the pool.

## Plots

```python
from sparc import plotting

plotting.plot_fuzzy_attribution(explanation, instance=0)   # membership functions of the fuzzy attributions
plotting.plot_linguistic_bars(explanation, instance=0)     # bars with support whiskers and labels
plotting.plot_pareto_front(explanation.front)              # fidelity loss against complexity, knee marked
plotting.plot_membership(explanation.partition)            # membership heat map
```

A plotting API in the style of `shap.plots` (bar, waterfall, beeswarm, heatmap, dependence, force, decision) and the views that only SPARC can draw (interval bars, linguistic profile, model disagreement, concept graph, front navigator, selection membership) are the next release; see `docs/roadmap.md`.

## Estimator options

- `point_estimate`: `"reference"` reports the attribution of the reference model on the original data beside the fuzzy number of the pool (default); `"centroid"` reports the centroid of the trapezoid.
- `concept_estimator`: `"aggregate"` forms the concept attribution from feature level Shapley values; `"direct"` computes Shapley values over the fuzzy concepts themselves (`sparc.concept_shapley`), with `direct_imputation` `"conditional"` or `"marginal"`.
- `crisp_concepts`, `partition_source`, `consensus`, `knee_method`, `alpha_grid`, `n_concepts_grid`, `fuzzifier_grid`: see the docstring of `sparc.Explainer`.

## Properties

The concept attribution satisfies efficiency for every membership matrix whose rows sum to one, reduces to SHAP for the identity partition, is symmetric, and is invariant to the duplication of a feature at the level of the concept that absorbs the copy. The direct estimator shows that the attribution is the Shapley value of a cooperative game over fuzzy concepts whose coalitions keep features to the degree of their membership. The tests under `tests/` check these properties numerically.

## The study

The evaluation of the method on synthetic problems with known attributions, on 15 public benchmarks and on open access datasets with known feature groups, together with the baselines, the metrics and the ablation analysis, lives in the study repository: https://github.com/charis-ntak/pfca.

## Citation

See `CITATION.cff`. A paper describing the method is in preparation.

## License

MIT.
