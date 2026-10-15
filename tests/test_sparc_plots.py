"""The views only SPARC can draw, and the navigation of the front through Explanation.at."""

import numpy as np
import pytest
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

import sparc

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from sparc import plots  # noqa: E402


@pytest.fixture(scope="module")
def explanation(small_problem):
    p, X_test = small_problem
    ex = sparc.Explainer(
        model_classes=[("gbm", GradientBoostingRegressor(n_estimators=40, random_state=0)), ("rf", RandomForestRegressor(n_estimators=30, random_state=0))],
        n_resamples=4, background_size=40, n_concepts_grid=(2, 3, 4), fuzzifier_grid=(2.0,), alpha_grid=(0.0, 0.5), disagreement_threshold=0.0, random_state=0,
    )
    ex.fit(p.X, p.y, feature_names=[f"f{j}" for j in range(p.n_features)])
    return ex(X_test[:14])


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def test_at_rebuilds_another_front_member(explanation):
    members = explanation.front_members()
    assert explanation.selection.knee in members and explanation.fidelity_optimal in members
    assert explanation.configuration_index == explanation.selection.knee
    other = explanation.fidelity_optimal
    e2 = explanation.at(other)
    assert e2.configuration_index == other
    assert e2.configuration.as_dict() == explanation.selection.configurations[other].as_dict()
    assert e2.n_explain == explanation.n_explain
    row = explanation.configurations.loc[other]
    assert row["fidelity_loss"] <= explanation.configurations.loc[explanation.selection.knee]["fidelity_loss"] + 1e-12
    # the same configuration gives the same values back
    same = explanation.at(explanation.selection.knee)
    np.testing.assert_allclose(same.values, explanation.values)
    # subsets are preserved through at()
    sub = explanation[[1, 3]].at(other)
    assert sub.n_explain == 2 and sub.instance_index.tolist() == [1, 3]
    np.testing.assert_allclose(sub.values, e2.values[[1, 3]])


def test_interval_bars_and_linguistic_profile(explanation):
    ax = plots.interval_bars(explanation[2])
    assert "instance 2" in ax.get_xlabel()
    assert len(ax.patches) == explanation.n_concepts
    ax2 = plots.interval_bars(explanation, instance=0, max_display=2)
    assert len(ax2.patches) == 2
    ax3 = plots.linguistic_profile(explanation[0])
    img = np.asarray(ax3.images[0].get_array())
    assert img.shape == (explanation.n_concepts, len(explanation.label_set.names))
    assert img.min() >= 0.0 and img.max() <= 1.0
    with pytest.raises(ValueError):
        plots.linguistic_profile(explanation)


def test_disagreement_instance_and_global(explanation):
    ax = plots.disagreement(explanation, instance=1, max_display=2)
    labels = [t.get_text() for t in ax.get_yticklabels()]
    assert len(labels) == 2 and all("disagreement" in label for label in labels)
    assert len(ax.get_legend().get_texts()) == 2
    ax2 = plots.disagreement(explanation)
    assert len(ax2.get_yticklabels()) == explanation.n_concepts
    assert ax2.get_xlabel().startswith("mean |attribution|")


def test_concept_graph(explanation):
    ax = plots.concept_graph(explanation)
    assert len([c for c in ax.collections]) >= explanation.partition.n_features
    texts = [t.get_text() for t in ax.texts]
    for name in explanation.feature_names:
        assert name in texts
    for name in explanation.concept_names:
        assert name in texts
    ax2 = plots.concept_graph(explanation, min_correlation=0.99)
    assert len(ax2.lines) <= len(ax.lines)


def test_front_and_navigator(explanation):
    ax = plots.front(explanation)
    legend = [t.get_text() for t in ax.get_legend().get_texts()]
    assert "knee (default)" in legend and "fidelity optimal" in legend and "this explanation" not in legend
    other = explanation.fidelity_optimal
    if other != explanation.selection.knee:
        ax2 = plots.front(explanation.at(other))
        assert "this explanation" in [t.get_text() for t in ax2.get_legend().get_texts()]
    axes = plots.front_navigator(explanation, instance=0, n_members=3)
    assert 1 <= len(axes) <= 3
    titles = [a.get_title() for a in axes]
    assert any("knee" in t for t in titles) and any("fidelity optimal" in t for t in titles)
    axes2 = plots.front_navigator(explanation, n_members=2)
    assert all("fidelity loss" in a.get_title() for a in axes2)
    assert plots.pareto_front(explanation.configurations) is not None


def test_selection_membership_and_coverage(explanation):
    axs = plots.selection_membership(explanation)
    assert len(axs) == 2
    assert len(axs[0].patches) == explanation.n_concepts
    assert len(axs[1].patches) == explanation.partition.n_features
    single = plots.selection_membership(explanation, features=False)
    assert single.get_xlabel().startswith("concept selection membership")
    # a replicate at the feature level and one at the concept level
    rng = np.random.default_rng(0)
    rep_feature = explanation.feature_attribution() + rng.normal(scale=1e-3, size=(explanation.n_explain, explanation.partition.n_features))
    ax = plots.coverage(explanation, rep_feature)
    assert len(ax.patches) == 2 * explanation.n_concepts
    rep_concept = explanation.values
    ax2 = plots.coverage(explanation, rep_concept)
    widths = sorted(p.get_width() for p in ax2.patches)
    assert 0.0 <= widths[0] and widths[-1] <= 1.0
    with pytest.raises(ValueError):
        plots.coverage(explanation, np.zeros((3, 3)))
