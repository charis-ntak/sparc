"""Tutorial 2: binary classification with known feature groups (Wisconsin breast cancer, built into scikit-learn).

Every measured characteristic of the cell nuclei appears three times, as a mean, a standard error and a worst
value, so the thirty features form ten natural groups of three strongly correlated features. The groups enter
SPARC as a candidate a priori partition next to the data driven ones, and the probability of the malignant
class is explained. Run with ``python examples/classification_breast_cancer.py [output_dir]``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.datasets import load_breast_cancer  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

import sparc  # noqa: E402
from sparc import plots  # noqa: E402

out = Path(sys.argv[1] if len(sys.argv) > 1 else "docs/assets/tutorial_classification")
out.mkdir(parents=True, exist_ok=True)

# --8<-- [start:data]
data = load_breast_cancer()
X, y, names = data.data, data.target, [n.replace(" ", "_") for n in data.feature_names]
# the a priori groups: the ten characteristics, each measured as mean, error and worst
groups = {}
for j, n in enumerate(names):
    key = n.replace("mean_", "").replace("worst_", "").replace("_error", "")
    groups.setdefault(key, []).append(j)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=0)
# --8<-- [end:data]

# --8<-- [start:fit]
explainer = sparc.Explainer(
    n_resamples=20,
    n_concepts_grid=(2, 3, 4, 5, 6),
    apriori_groups=groups,          # domain knowledge enters as a candidate partition
    partition_source="both",
    target_class=0,                 # classes_ are [0, 1] with 0 = malignant in this dataset
    random_state=0,
    n_jobs=4,
)
explainer.fit(X_train, y_train, feature_names=names)
explanation = explainer(X_test[:80])
print(explanation)
print("explained class:", explanation.attribution.explained_class, "of", explanation.attribution.classes)
print(explanation.describe(0))
# --8<-- [end:fit]


def save(ax, name):
    fig = ax.figure if hasattr(ax, "figure") else ax[0].figure
    fig.tight_layout()
    fig.savefig(out / f"{name}.png", dpi=120)
    plt.close("all")


# --8<-- [start:plots]
save(plots.bar(explanation), "bar")
save(plots.membership(explanation.partition), "membership")
save(plots.concept_graph(explanation, min_correlation=0.6), "concept_graph")
save(plots.beeswarm(explanation), "beeswarm")
save(plots.waterfall(explanation[0]), "waterfall")
save(plots.interval_bars(explanation[0]), "interval_bars")
save(plots.disagreement(explanation[0]), "disagreement")
save(plots.front(explanation), "front")
save(plots.selection_membership(explanation), "selection_membership")
# --8<-- [end:plots]

# --8<-- [start:html]
(out / "card.html").write_text(explanation.as_html(0), encoding="utf8")
print(explanation.as_list(0, retained_only=True))
# --8<-- [end:html]
print("figures written to", out)
