# Changelog

## 0.1.2 (2026-10-15)

- The logo, on the project page and in the documentation. No code change.

## 0.1.1 (2026-10-15)

- First release available on PyPI. Identical in content to 0.1.0, whose files were withdrawn from PyPI the same day; version numbers cannot be reused there.

## 0.1.0 (2026-10-15)

- First release of the package, moved from the study repository charis-ntak/pfca: the attribution pool, the fuzzy concepts, the fuzzification with linguistic labels, the Pareto selection, the explainer and the existing plots.
- Estimator options: reference centred point estimate (default), direct Shapley values over fuzzy concepts with conditional or marginal imputation, importance weighted redistribution to features.
- Synthetic problem generators for examples and tests.
- Documentation site (MkDocs) with getting started, three tutorials run from `examples/` on built in datasets, a plot gallery, the estimator options, the properties and the API reference, deployed to GitHub Pages from CI.
- Array interface of the explanation: `values`, `base_values`, `data` (concept scores), `names`, `importance`, indexing by instance; the views `as_list`, `as_dataframe`, `as_html`, `show_in_notebook`; `explainer(X)` as a shorthand of `explain`.
- `sparc.plots`: bar, waterfall, beeswarm, heatmap, dependence, force and decision at the concept level, beside the fuzzy attribution, linguistic bar, Pareto front and membership plots; the views only SPARC can draw: interval bars, linguistic profile, disagreement, concept graph, front, front navigator, selection membership and coverage.
- `Explanation.at(index)`, `fidelity_optimal`, `front_members()`, `configurations` and `configuration_index`, to navigate the front; subsets of an explanation slice the partition pools so that they can be rebuilt at other members.
