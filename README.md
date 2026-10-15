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

explainer = sparc.Explainer(n_resamples=20, random_state=0)
explainer.fit(X_train, y_train, feature_names=[f"x{j}" for j in range(9)])
explanation = explainer(X_test[:20])   # or explainer.explain(...)

explanation.values                      # point concept attribution, (n_explain, K)
explanation.base_values                 # value the attributions are measured from
explanation.names                       # concept names
explanation[0]                          # the explanation of one instance (slices and index lists work too)
explanation.as_list(0)                  # (concept, value, label, sign confidence), most important first
print(explanation.describe(0))          # the linguistic explanation of the first instance
explanation.instance_frame(0)           # concepts, fuzzy attribution, sign confidence, labels, as a table
explanation.as_dataframe()              # the long table of every instance and concept
explanation.show_in_notebook(0)         # an HTML card in a Jupyter notebook
explanation.front                       # the Pareto front of explanation configurations
explanation.membership                  # feature by concept membership matrix
```


The default model classes are a gradient boosting model (the reference model), a random forest and a multilayer perceptron; pass `model_classes=[("gbm", GradientBoostingRegressor()), ...]` to change them. Classifiers are explained through the probability of `target_class`. `apriori_groups` adds a known grouping of the features as a candidate partition, `solver="nsga2"` searches large decision spaces, and `n_jobs` parallelizes the pool.

The explanation exposes `values`, `base_values`, `data` (concept scores), `names` and `importance`, can be indexed by instance, and offers `as_list`, `as_dataframe`, `as_html` and `show_in_notebook`, so the usual workflows of attribution libraries carry over without translation.

## Plots

Every plot takes an explanation, or one instance of it, draws with matplotlib and returns the axes.

```python
from sparc import plots

plots.bar(explanation)                      # concept importance with the support of the fuzzy global importance
plots.bar(explanation[0])                   # signed attributions of one instance with their supports
plots.waterfall(explanation[0])             # from the base value to the output, concept by concept
plots.beeswarm(explanation)                 # one dot per instance and concept, coloured by the concept score
plots.heatmap(explanation)                  # instances by concepts, ordered by the model output
plots.dependence(explanation, "C1")         # attribution against the concept score, coloured by the interacting concept
plots.force(explanation[0])                 # contributions pushing the output away from the base value
plots.decision(explanation)                 # cumulative paths of all instances through the concepts

```

The views that only SPARC can draw, because it carries a distribution per attribution, the disagreement between model classes, a partition with degrees of membership and a front of explanations:

```python
plots.interval_bars(explanation[0])         # point, core and support of every attribution, sign confidence, disagreement
plots.linguistic_profile(explanation[0])    # degree of every linguistic label for every concept
plots.disagreement(explanation[0])          # the attribution distribution of every model class, concept by concept
plots.concept_graph(explanation)            # the fuzzy partition as a picture of the features
plots.front(explanation)                    # the Pareto front with the knee and the fidelity optimal member
plots.front_navigator(explanation, 0)       # the explanation along the front, member by member
plots.selection_membership(explanation)     # how often the front retains every concept and feature
plots.coverage(explanation, replicate)      # coverage of a replicated attribution by the intervals

plots.fuzzy_attribution(explanation, 0)     # membership functions of the fuzzy attributions
plots.linguistic_bars(explanation, 0)       # bars with support whiskers and linguistic labels
plots.membership(explanation.partition)     # feature by concept membership heat map
```

Retained concepts are marked with an asterisk in the axis labels. `explanation.at(index)` rebuilds the explanation at any other evaluated configuration, `explanation.fidelity_optimal` is the most faithful member of the front and `explanation.front_members()` lists the front from the simplest to the most complex explanation, so the knee is a default, not a limit.

## Estimator options

- `point_estimate`: `"reference"` reports the attribution of the reference model on the original data beside the fuzzy number of the pool (default); `"centroid"` reports the centroid of the trapezoid.
- `concept_estimator`: `"aggregate"` forms the concept attribution from feature level Shapley values; `"direct"` computes Shapley values over the fuzzy concepts themselves (`sparc.concept_shapley`), with `direct_imputation` `"conditional"` or `"marginal"`.
- `crisp_concepts`, `partition_source`, `consensus`, `knee_method`, `alpha_grid`, `n_concepts_grid`, `fuzzifier_grid`: see the docstring of `sparc.Explainer`.

## Properties

The concept attribution satisfies efficiency for every membership matrix whose rows sum to one, reduces to feature level Shapley values for the identity partition, is symmetric, and is invariant to the duplication of a feature at the level of the concept that absorbs the copy. The direct estimator shows that the attribution is the Shapley value of a cooperative game over fuzzy concepts whose coalitions keep features to the degree of their membership. The tests under `tests/` check these properties numerically.

## The study

The evaluation of the method on synthetic problems with known attributions, on 15 public benchmarks and on open access datasets with known feature groups, together with the baselines, the metrics and the ablation analysis, lives in the study repository: https://github.com/charis-ntak/pfca.

## Citation

See `CITATION.cff`. A paper describing the method is in preparation.

## License

MIT.
