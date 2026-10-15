"""The plots of sparc.plots: every function draws from an explanation and returns the axes."""

import numpy as np
import pytest
from sklearn.ensemble import GradientBoostingRegressor

import sparc

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from sparc import plots  # noqa: E402


@pytest.fixture(scope="module")
def explanation(small_problem):
    p, X_test = small_problem
    ex = sparc.Explainer(model_classes=[("gbm", GradientBoostingRegressor(n_estimators=40, random_state=0))], n_resamples=4, background_size=40, n_concepts_grid=(2, 3), fuzzifier_grid=(2.0,), alpha_grid=(0.0, 0.5), random_state=0)
    ex.fit(p.X, p.y, feature_names=[f"f{j}" for j in range(p.n_features)])
    return ex(X_test[:16])


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def test_bar_global_and_instance(explanation):
    ax = plots.bar(explanation)
    labels = [t.get_text() for t in ax.get_yticklabels()]
    assert len(labels) == explanation.n_concepts
    order = np.argsort(-explanation.importance, kind="stable")
    assert labels[0].rstrip(" *") == explanation.concept_names[order[0]]
    assert any(label.endswith("*") for label in labels)  # retained concepts are marked
    ax2 = plots.bar(explanation, instance=2, max_display=2)
    assert len(ax2.get_yticklabels()) == min(2, explanation.n_concepts)
    ax3 = plots.bar(explanation[5])
    assert "instance 5" in ax3.get_xlabel()


def test_waterfall_ends_at_the_output(explanation):
    e = explanation[3]
    ax = plots.waterfall(e)
    patches = [p for p in ax.patches]
    assert len(patches) == e.n_concepts
    end = float(e.base_values[0] + sum(p.get_width() for p in patches))
    assert end == pytest.approx(float(e.output[0]), abs=1e-6)
    ax2 = plots.waterfall(explanation, instance=0, max_display=1)
    assert len(ax2.patches) == 2 and "other" in ax2.get_yticklabels()[0].get_text() or len(ax2.patches) == 2
    with pytest.raises(ValueError):
        plots.waterfall(explanation)


def test_beeswarm_and_heatmap(explanation):
    ax = plots.beeswarm(explanation, max_display=3)
    assert len(ax.collections) >= min(3, explanation.n_concepts)
    assert len(ax.get_yticklabels()) == min(3, explanation.n_concepts)
    ax2 = plots.heatmap(explanation)
    img = ax2.images[0].get_array()
    assert img.shape == (explanation.n_concepts, explanation.n_explain)
    order = np.argsort(explanation.output, kind="stable")
    top = np.argsort(-explanation.importance, kind="stable")
    np.testing.assert_allclose(np.asarray(img)[0], explanation.values[order, top[0]])
    ax3 = plots.heatmap(explanation, instance_order="none")
    np.testing.assert_allclose(np.asarray(ax3.images[0].get_array())[0], explanation.values[:, top[0]])
    with pytest.raises(ValueError):
        plots.heatmap(explanation, instance_order="other")


def test_dependence_by_name_and_index(explanation):
    name = explanation.concept_names[0]
    ax = plots.dependence(explanation, name)
    assert name in ax.get_xlabel() and name in ax.get_ylabel()
    xy = ax.collections[0].get_offsets()
    np.testing.assert_allclose(np.asarray(xy)[:, 1], explanation.values[:, 0])
    ax2 = plots.dependence(explanation, 1, color=0)
    assert ax2.figure.axes[-1].get_ylabel().endswith(explanation.concept_names[0])
    ax3 = plots.dependence(explanation, 0, color=False)
    assert len(ax3.figure.axes) == 1
    with pytest.raises(KeyError):
        plots.dependence(explanation, "nope")
    with pytest.raises(IndexError):
        plots.dependence(explanation, 99)


def test_force_and_decision(explanation):
    e = explanation[1]
    ax = plots.force(e)
    widths = sum(p.get_width() for p in ax.patches)
    assert widths == pytest.approx(float(np.abs(e.values[0]).sum()), abs=1e-6)
    assert "instance 1" in ax.get_xlabel()
    ax2 = plots.decision(explanation, max_display=2)
    assert len(ax2.lines) >= explanation.n_explain
    labels = [t.get_text() for t in ax2.get_yticklabels()]
    assert labels[0].startswith("base") and len(labels) == min(2, explanation.n_concepts) + 1
    ax3 = plots.decision(explanation, instances=[0, 1, 2])
    assert len([ln for ln in ax3.lines if len(ln.get_xdata()) > 1]) >= 3


def test_legacy_plots_are_reexported(explanation):
    assert plots.fuzzy_attribution(explanation, 0) is not None
    assert plots.linguistic_bars(explanation, 0) is not None
    assert plots.pareto_front(explanation.front) is not None
    assert plots.membership(explanation.partition) is not None
    assert plots.bar(explanation, show=False) is not None
