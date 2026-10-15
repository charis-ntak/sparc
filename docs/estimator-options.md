# Estimator options

The constructor of `sparc.Explainer` exposes the design choices of the method. The defaults follow the ablation analysis of the study; the options below are the ones worth knowing.

## The pool

| Option | Default | Meaning |
|---|---|---|
| `model_classes` | gradient boosting, random forest, multilayer perceptron | The model classes of the pool; the first is the reference model whose output is explained. Pass `[("name", estimator), ...]`. |
| `n_resamples` | 50 | Bootstrap resamples B; resample 0 is the original data. |
| `background_size` | 100 | Background sample of the Shapley computation, drawn from the training rows of every pool member. |
| `explainer` | `"auto"` | `"tree"`, `"linear"`, `"exact"`, `"permutation"` or `"kernel"`; `"auto"` picks the exact tree estimator for tree ensembles, the exact estimator up to ten features and permutation sampling above. |
| `task`, `target_class` | `"auto"`, 1 | Regression or classification, and the class whose probability is explained. |

The pool over model classes is what makes the intervals honest: in the ablation study a single model class kept the stability but lost the calibration of the intervals (support coverage fell from 0.92 to 0.99 down to 0.78 to 0.95 across datasets).

## The concepts

| Option | Default | Meaning |
|---|---|---|
| `n_concepts_grid`, `fuzzifier_grid` | 2 to 6, 1.5 to 2.5 | Candidate numbers of concepts and fuzzifiers of fuzzy c means. |
| `concept_distance` | `"correlation"` | Features are clustered by their correlation profiles; `"loading"` uses factor loadings instead. |
| `consensus` | False | Consensus clustering over resamples of the instances, more stable partitions at a cost. |
| `apriori_groups`, `partition_source` | None, `"both"` | A known grouping of the features as a candidate partition; `"data"`, `"apriori"` or `"both"`. |
| `crisp_concepts` | False | Ablation: harden the partitions. Crisp concepts keep the stability, and in the ablation they removed the output faster on one dataset because fuzzy removal is partial. |

## The fuzzy numbers

| Option | Default | Meaning |
|---|---|---|
| `fuzzy_shape` | `"trapezoidal"` | Or `"triangular"`. |
| `support_percentiles`, `core_percentiles` | (5, 95), (25, 75) | The quantiles of the pool that form the support and the core. |
| `disagreement_threshold` | 0.5 | Above this disagreement index an attribution is upgraded to an interval type 2 number, one trapezoid per model class. |
| `linguistic_labels`, `linguistic_scale` | five labels, `"mean_abs"` | The label set and the standardization of the attribution axis. |

## The selection

| Option | Default | Meaning |
|---|---|---|
| `alpha_grid`, `sparsity_grid` | 0 to 1 in steps of 0.25, 1 to K | Alpha levels of the cut and numbers of retained concepts evaluated. |
| `solver` | `"grid"` | Exhaustive evaluation; `"nsga2"` for large decision spaces; `"fixed"` evaluates `fixed_configuration` only (ablation). |
| `knee_method` | `"utopia"` | The default explanation: the front member closest to the utopia point; `"hyperplane"` is the alternative. |
| `surrogate`, `cv` | `"linear"`, 5 | The surrogate and the folds of the fidelity loss. |

The selection is what buys stability and sparsity: without it (five concepts, all retained) the surrogate fit is perfect and the insertion area higher, but the deletion area is much worse and the rank stability falls to 0.46 to 0.75. The price of the knee is a lower insertion area, which is why the front, the fidelity optimal member and `explanation.at()` are part of the interface.

## The point estimate and the estimator

| Option | Default | Meaning |
|---|---|---|
| `point_estimate` | `"reference"` | The reported number is the attribution of the reference model on the original data, while the fuzzy number, the sign confidence and the disagreement describe the whole pool. `"centroid"` reports the centroid of the trapezoid instead. |
| `concept_estimator` | `"aggregate"` | The concept attribution is the membership weighted sum of feature level Shapley values. `"direct"` computes Shapley values over the fuzzy concepts of the selected partition directly (`sparc.concept_shapley`): every concept is a player and a coalition keeps features to the degree of their membership. |
| `direct_imputation` | `"conditional"` | How the direct estimator removes features: by the mean of the nearest background rows on the kept features, or `"marginal"`, the mean over the background sample. |
| `feature_weights` | `"membership"` | How a concept's credit is redistributed to its features by `feature_attribution`; `"importance"` weights the members by their mean absolute feature level attribution. |

What the ablation found. The reference centred point raises the fit of the reported attribution to the model on every dataset (surrogate R² from 0.87 to 0.97 on one of them) and changes nothing else, hence the default. The direct estimator with marginal imputation reproduces the aggregated one within 0.01 on every metric: the interventional concept game and the membership aggregation of interventional feature values agree, which is a useful check of the method's foundation. The direct estimator with conditional imputation anticipates the removal used by deletion tests and improves the deletion area on some datasets, but it pays with rank stability, so it is an option and an ablation, not the default.
