"""Tutorial 1: a regression problem with a known concept structure.

Three latent factors, each measured by three correlated features, drive a nonlinear response. SPARC should
recover the three blocks as concepts, attribute the prediction to them with calibrated fuzzy numbers and
report a stable ranking. Run with ``python examples/regression_synthetic.py [output_dir]``; the figures are
written to the output directory (default ``docs/assets/tutorial_regression``).
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

import sparc  # noqa: E402
from sparc import plots  # noqa: E402
from sparc.synthetic import make_synthetic  # noqa: E402

out = Path(sys.argv[1] if len(sys.argv) > 1 else "docs/assets/tutorial_regression")
out.mkdir(parents=True, exist_ok=True)

# --8<-- [start:data]
problem = make_synthetic("additive", n_samples=800, n_features=9, rho=0.8, n_factors=3, random_state=0)
X_train, X_test, y_train, y_test = train_test_split(problem.X, problem.y, test_size=0.25, random_state=0)
feature_names = [f"x{j}" for j in range(9)]
# --8<-- [end:data]

# --8<-- [start:fit]
explainer = sparc.Explainer(n_resamples=20, n_concepts_grid=(2, 3, 4, 5), random_state=0, n_jobs=4)
explainer.fit(X_train, y_train, feature_names=feature_names)
explanation = explainer(X_test[:60])
print(explanation)
print(explanation.describe(0))
# --8<-- [end:fit]

# --8<-- [start:tables]
print(explanation.instance_frame(0).round(3).to_string(index=False))
print(explanation.global_frame().round(3).to_string(index=False))
print(explanation.front[["n_concepts", "fuzzifier", "sparsity", "alpha", "fidelity_loss", "complexity", "instability", "knee"]].round(3).to_string())
# --8<-- [end:tables]


def save(ax, name):
    fig = ax.figure if hasattr(ax, "figure") else ax[0].figure
    fig.tight_layout()
    fig.savefig(out / f"{name}.png", dpi=120)
    plt.close("all")


# --8<-- [start:plots]
save(plots.bar(explanation), "bar")
save(plots.beeswarm(explanation), "beeswarm")
save(plots.waterfall(explanation[0]), "waterfall")
save(plots.interval_bars(explanation[0]), "interval_bars")
save(plots.fuzzy_attribution(explanation, 0), "fuzzy_attribution")
save(plots.linguistic_profile(explanation[0]), "linguistic_profile")
save(plots.disagreement(explanation[0]), "disagreement")
save(plots.membership(explanation.partition), "membership")
save(plots.concept_graph(explanation), "concept_graph")
save(plots.front(explanation), "front")
save(plots.front_navigator(explanation, instance=0, n_members=3), "front_navigator")
save(plots.selection_membership(explanation), "selection_membership")
save(plots.heatmap(explanation), "heatmap")
save(plots.dependence(explanation, 0), "dependence")
save(plots.force(explanation[0]), "force")
save(plots.decision(explanation), "decision")
# --8<-- [end:plots]

# --8<-- [start:coverage]
# a replicate: the same instances explained by an explainer fitted on another split, scored against the intervals
X_train2, _, y_train2, _ = train_test_split(problem.X, problem.y, test_size=0.25, random_state=1)
replica = sparc.Explainer(n_resamples=20, n_concepts_grid=(2, 3, 4, 5), random_state=1, n_jobs=4).fit(X_train2, y_train2, feature_names=feature_names)
replicate = replica(X_test[:60]).feature_attribution()
save(plots.coverage(explanation, replicate), "coverage")
# --8<-- [end:coverage]

# --8<-- [start:truth]
# the true concept structure is known here: compare the recovered partition with it
from sklearn.metrics import adjusted_rand_score  # noqa: E402

print("adjusted Rand index of the hardened partition with the true blocks:", round(adjusted_rand_score(problem.true_membership.argmax(axis=1), explanation.partition.labels), 3))
print("sum of concept values + base value = model output:", np.allclose(explanation.values.sum(axis=1) + explanation.base_values, explanation.output, atol=1e-6))
# --8<-- [end:truth]
print("figures written to", out)
