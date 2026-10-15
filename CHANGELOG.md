# Changelog

## 0.1.0 (unreleased)

- First release of the package, moved from the study repository charis-ntak/pfca: the attribution pool, the fuzzy concepts, the fuzzification with linguistic labels, the Pareto selection, the explainer and the existing plots.
- Estimator options: reference centred point estimate (default), direct Shapley values over fuzzy concepts with conditional or marginal imputation, importance weighted redistribution to features.
- Synthetic problem generators for examples and tests.
- Array interface of the explanation: `values`, `base_values`, `data` (concept scores), `names`, `importance`, indexing by instance; the views `as_list`, `as_dataframe`, `as_html`, `show_in_notebook`; `explainer(X)` as a shorthand of `explain`.
- `sparc.plots`: bar, waterfall, beeswarm, heatmap, dependence, force and decision at the concept level, beside the fuzzy attribution, linguistic bar, Pareto front and membership plots.
