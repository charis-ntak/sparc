"""Tutorial 3: a problem with more than two classes (wine recognition, built into scikit-learn).

A classifier with three classes is explained one class at a time: the explainer takes ``target_class`` and
explains the predicted probability of that class, so that every method of the package applies unchanged. The
three explanations of the same instances are compared side by side. Run with
``python examples/multiclass_wine.py [output_dir]``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.datasets import load_wine  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

import sparc  # noqa: E402
from sparc import plots  # noqa: E402

out = Path(sys.argv[1] if len(sys.argv) > 1 else "docs/assets/tutorial_multiclass")
out.mkdir(parents=True, exist_ok=True)

# --8<-- [start:data]
data = load_wine()
X, y, names = data.data, data.target, list(data.feature_names)
classes = list(data.target_names)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)
# --8<-- [end:data]

# --8<-- [start:fit]
explanations = {}
for k, class_name in enumerate(classes):
    explainer = sparc.Explainer(n_resamples=20, n_concepts_grid=(2, 3, 4, 5), target_class=k, random_state=0, n_jobs=4)
    explainer.fit(X_train, y_train, feature_names=names)
    explanations[class_name] = explainer(X_test[:40])
    print(class_name, explanations[class_name])
# --8<-- [end:fit]


def save(ax, name):
    fig = ax.figure if hasattr(ax, "figure") else ax[0].figure
    fig.tight_layout()
    fig.savefig(out / f"{name}.png", dpi=120)
    plt.close("all")


# --8<-- [start:plots]
fig, axes = plt.subplots(3, 1, figsize=(9, 8.5))
for ax, (class_name, e) in zip(axes, explanations.items()):
    plots.bar(e, ax=ax)
    ax.set_title(f"probability of class {class_name}", fontsize=10)
fig.tight_layout()
fig.savefig(out / "bar_by_class.png", dpi=120)
plt.close("all")

# one instance, three classes: which concepts push each class probability
i = 0
fig, axes = plt.subplots(3, 1, figsize=(9, 8.5))
for ax, (class_name, e) in zip(axes, explanations.items()):
    plots.interval_bars(e[i], ax=ax, compact=True)
    ax.set_title(f"instance {i}, P({class_name}) = {e.output[i]:.2f}", fontsize=10)
fig.tight_layout()
fig.savefig(out / "instance_by_class.png", dpi=120)
plt.close("all")

e = explanations[classes[0]]
save(plots.membership(e.partition), "membership")
save(plots.linguistic_profile(e[i]), "linguistic_profile")
save(plots.heatmap(e), "heatmap")
# --8<-- [end:plots]
print("figures written to", out)
