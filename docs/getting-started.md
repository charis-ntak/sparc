# Getting started

## Install

```
pip install "sparc-xai[plots]"
```

Python 3.10 or later. The package is imported as `sparc`. Without the `plots` extra everything works except `sparc.plots`.

## Fit, explain, read

```python
import sparc
from sparc.synthetic import make_synthetic
from sklearn.model_selection import train_test_split

problem = make_synthetic("additive", n_samples=800, n_features=9, rho=0.8, n_factors=3, random_state=0)
X_train, X_test, y_train, y_test = train_test_split(problem.X, problem.y, random_state=0)

explainer = sparc.Explainer(n_resamples=20, random_state=0)
explainer.fit(X_train, y_train, feature_names=[f"x{j}" for j in range(9)])
explanation = explainer(X_test[:50])

print(explanation)
print(explanation.describe(0))
```

`Explainer.fit` trains the pool: every model class on the original data and on `n_resamples - 1` bootstrap resamples. The default model classes are a gradient boosting model, which is the reference model, a random forest and a multilayer perceptron; a classifier is explained through the probability of `target_class`. `fit` also forms the candidate concept partitions by fuzzy c means on the correlation profiles of the features, for every number of concepts in `n_concepts_grid` and every fuzzifier in `fuzzifier_grid`.

`explainer(X)` computes the feature level Shapley values of every pool member for the instances of `X`, aggregates them to every candidate partition, evaluates every configuration (partition, sparsity, alpha level) on the three objectives, keeps the Pareto front and returns the explanation at the knee.

The sentence form of one instance reads like this:

```
Instance 0: model output 0.773
Configuration: K=3, fuzzifier=2.0, sparsity=2, alpha=0.00
  C2 (x3, x4, x5): weakly positive (compatibility 1.00), centroid +1.05, support [+0.79, +1.32], core [+0.9, +1.15], sign confidence 1.00, ...
  C1 (x0, x1, x2): negligible (compatibility 0.90), centroid +0.273, support [-0.33, +0.86], ...
```

## The explanation object

| Attribute or method | Meaning |
|---|---|
| `values` | point attribution of every concept, shape (instances, concepts) |
| `base_values`, `output` | the value the attributions are measured from, and the model output; `values.sum(1) + base_values == output` |
| `names`, `feature_names` | concept names and feature names |
| `importance` | mean absolute attribution of every concept |
| `data` | concept scores of the instances (standardized features averaged with the memberships), used to colour plots |
| `retained` | indices of the concepts the knee explanation retains |
| `fuzzy(i, k)` | the trapezoidal fuzzy number of instance `i` and concept `k`, or an interval type 2 number when the model classes disagree |
| `sign_confidence`, `disagreement` | per instance and concept |
| `labels(i)`, `describe(i)` | linguistic labels and the sentence form |
| `partition`, `membership` | the fuzzy partition and its membership matrix |
| `front`, `configurations` | the Pareto front, and every evaluated configuration |
| `at(index)`, `fidelity_optimal`, `front_members()` | another member of the front, the most faithful one, and the whole front in order |
| `instance_frame(i)`, `global_frame()`, `as_dataframe()`, `as_list(i)`, `as_html(i)`, `show_in_notebook(i)` | tables and views |
| `explanation[i]`, `explanation[2:5]`, `explanation[[0, 3]]` | subsets of the explained instances, with every per instance quantity sliced |

## Plot

```python
from sparc import plots

plots.bar(explanation)                  # concept importance with the support of the fuzzy global importance
plots.interval_bars(explanation[0])     # one instance: point, core, support, sign confidence, disagreement
plots.front(explanation)                # the Pareto front with the knee and the fidelity optimal member
plots.front_navigator(explanation, 0)   # the same instance along the front
```

Every plot returns a matplotlib axes; pass `ax=` to draw into your own figure and `show=True` to display at once. See the [gallery](gallery.md).

## Your own models

```python
from sklearn.ensemble import HistGradientBoostingRegressor, ExtraTreesRegressor
from sklearn.neural_network import MLPRegressor

explainer = sparc.Explainer(model_classes=[
    ("hgb", HistGradientBoostingRegressor()),     # the first is the reference model
    ("et", ExtraTreesRegressor(n_estimators=200)),
    ("mlp", MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=500)),
])
```

Any scikit-learn estimator with `fit` and `predict` (or `predict_proba`) works; tree ensembles are explained exactly and fast, other models through the exact or the permutation estimator depending on the number of features. Known groups of features enter as a candidate partition with `apriori_groups={"group name": [column indices or names], ...}` and `partition_source="both"`.
