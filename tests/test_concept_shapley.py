"""Properties of the direct concept estimator (Shapley values over fuzzy concepts) and of the estimator options."""

import numpy as np
import pytest
from helpers import grouped_shapley
from sklearn.ensemble import GradientBoostingRegressor

from sparc import Explainer
from sparc.attribution import AttributionEngine, redistribute_to_features, shapley_values
from sparc.concept_shapley import concept_shapley_values, fuzzy_removal_weights
from sparc.utils import normalize_rows


def _nonlinear(X):
    X = np.asarray(X, dtype=float)
    return X[:, 0] * X[:, 1] + np.sin(X[:, 2]) + 0.5 * X[:, 3] ** 2 - X[:, 3] * X[:, 0]


class _Model:
    def predict(self, X):
        return _nonlinear(X)


def test_removal_weights_follow_the_membership_mass():
    U = np.array([[1.0, 0.0], [0.6, 0.4], [0.0, 1.0]])
    np.testing.assert_allclose(fuzzy_removal_weights(U, np.array([True, False])), [0.0, 0.4, 1.0])
    np.testing.assert_allclose(fuzzy_removal_weights(U, np.array([True, True])), [0.0, 0.0, 0.0])
    np.testing.assert_allclose(fuzzy_removal_weights(U, np.array([False, False])), [1.0, 1.0, 1.0])


@pytest.mark.parametrize("imputation", ["marginal", "conditional"])
def test_efficiency_for_fuzzy_partitions(imputation):
    rng = np.random.default_rng(0)
    bg = rng.normal(size=(60, 4))
    X = rng.normal(size=(5, 4))
    for K in (1, 2, 3):
        U = normalize_rows(rng.random((4, K)))
        values, empty = concept_shapley_values(_nonlinear, X, bg, U, imputation=imputation)
        np.testing.assert_allclose(values.sum(axis=1) + empty, _nonlinear(X), atol=1e-9)


def test_reduction_to_feature_shapley_values_with_the_identity_partition():
    rng = np.random.default_rng(1)
    bg = rng.normal(size=(40, 4))
    X = rng.normal(size=(3, 4))
    values, empty = concept_shapley_values(_nonlinear, X, bg, np.eye(4), imputation="marginal")
    sv, expected = shapley_values(_Model(), X, bg, "regression", explainer="exact")
    np.testing.assert_allclose(values, sv, atol=1e-8)
    assert empty[0] == pytest.approx(expected, abs=1e-8)


def test_reduction_to_grouped_shapley_values_for_crisp_groups():
    rng = np.random.default_rng(2)
    bg = rng.normal(size=(40, 4))
    X = rng.normal(size=(3, 4))
    U = np.array([[1, 0], [1, 0], [0, 1], [0, 1]], dtype=float)
    values, _ = concept_shapley_values(_nonlinear, X, bg, U, imputation="marginal")
    ref, _ = grouped_shapley(_nonlinear, X, bg, [[0, 1], [2, 3]])
    np.testing.assert_allclose(values, ref, atol=1e-8)


def test_symmetry_of_interchangeable_concepts(linear_model_cls):
    model = linear_model_cls([1.0, 1.0, 1.0, 1.0])
    rng = np.random.default_rng(0)
    bg = rng.normal(size=(200, 4))
    bg -= bg.mean(axis=0)
    x = np.array([[1.0, 2.0, 3.0, 0.0]])
    U = np.array([[1, 0], [1, 0], [0, 1], [0, 1]], dtype=float)
    values, _ = concept_shapley_values(model.predict, x, bg, U, imputation="marginal")
    assert values[0, 0] == pytest.approx(values[0, 1], abs=1e-9)


def test_duplication_invariance_of_the_direct_estimator(linear_model_cls):
    """An exact copy of a feature that joins its concept leaves every concept value unchanged."""
    rng = np.random.default_rng(1)
    bg = rng.normal(size=(300, 2))
    x = np.array([[1.5, -0.5]])
    before = linear_model_cls([2.0, 1.0])
    U = np.eye(2)
    v_before, _ = concept_shapley_values(before.predict, x, bg, U, imputation="marginal")
    bg2 = np.hstack([bg, bg[:, [0]]])
    x2 = np.array([[1.5, -0.5, 1.5]])
    after = linear_model_cls([1.0, 1.0, 1.0])
    U2 = np.array([[1, 0], [0, 1], [1, 0]], dtype=float)
    v_after, _ = concept_shapley_values(after.predict, x2, bg2, U2, imputation="marginal")
    np.testing.assert_allclose(v_after, v_before, atol=1e-9)


def test_permutation_sampling_matches_exact_values_for_small_games():
    rng = np.random.default_rng(3)
    bg = rng.normal(size=(30, 4))
    X = rng.normal(size=(2, 4))
    U = normalize_rows(rng.random((4, 3)))
    exact, e1 = concept_shapley_values(_nonlinear, X, bg, U, imputation="marginal")
    sampled, e2 = concept_shapley_values(_nonlinear, X, bg, U, imputation="marginal", max_exact_concepts=2, n_permutations=400, random_state=0)
    np.testing.assert_allclose(e1, e2)
    np.testing.assert_allclose(sampled, exact, atol=0.05 * np.abs(exact).max() + 1e-6)


def test_importance_weighted_redistribution_keeps_concept_totals():
    rng = np.random.default_rng(4)
    U = normalize_rows(rng.random((5, 2)))
    C = rng.normal(size=(3, 2))
    w = rng.random(5)
    plain = redistribute_to_features(C, U)
    weighted = redistribute_to_features(C, U, w)
    # the total credit of every instance is preserved by both redistributions
    np.testing.assert_allclose(plain.sum(axis=1), C.sum(axis=1), atol=1e-9)
    np.testing.assert_allclose(weighted.sum(axis=1), C.sum(axis=1), atol=1e-9)
    assert not np.allclose(plain, weighted)
    # with crisp concepts the members of a concept receive exactly its value, in both schemes
    Uc = np.array([[1, 0], [1, 0], [1, 0], [0, 1], [0, 1]], dtype=float)
    np.testing.assert_allclose(redistribute_to_features(C, Uc) @ Uc, C, atol=1e-9)
    np.testing.assert_allclose(redistribute_to_features(C, Uc, w) @ Uc, C, atol=1e-9)
    # identity partition: both reduce to the concept values
    np.testing.assert_allclose(redistribute_to_features(C[:, :1].repeat(5, axis=1), np.eye(5), w), C[:, :1].repeat(5, axis=1))


def test_engine_explain_concepts_is_efficient_for_every_member(small_problem):
    p, X_test = small_problem
    eng = AttributionEngine(model_classes=[("gbm", GradientBoostingRegressor(n_estimators=30, random_state=0))], n_resamples=2, background_size=30, random_state=0)
    res = eng.fit_explain(p.X, p.y, X_test[:4])
    U = normalize_rows(np.random.default_rng(0).random((p.n_features, 3)))
    pool, empty = eng.explain_concepts(X_test[:4], U, imputation="marginal")
    assert pool.shape == (2, 1, 4, 3) and empty.shape == (2, 1, 4)
    np.testing.assert_allclose(pool.sum(axis=-1) + empty, res.outputs, atol=1e-8)
    assert res.X_explain is not None and res.X_explain.shape == (4, p.n_features)


@pytest.mark.parametrize("imputation", ["conditional", "marginal"])
def test_direct_explainer_runs_and_keeps_the_selection(small_problem, imputation):
    p, X_test = small_problem
    common = dict(model_classes=[("gbm", GradientBoostingRegressor(n_estimators=30, random_state=0))], n_resamples=3, background_size=30, n_concepts_grid=(2, 3), fuzzifier_grid=(2.0,), alpha_grid=(0.0, 0.5), random_state=0)
    base = Explainer(**common).fit(p.X, p.y).explain(X_test[:8])
    direct = Explainer(concept_estimator="direct", direct_imputation=imputation, **common).fit(p.X, p.y).explain(X_test[:8])
    assert direct.concept_estimator == "direct"
    assert direct.configuration.as_dict() == base.configuration.as_dict()
    assert direct.concept_pool.shape == base.concept_pool.shape
    assert direct.empty_coalition_value is not None and direct.empty_coalition_value.shape == (3, 8)
    # efficiency of every pool member with respect to its empty coalition value
    outputs = direct.attribution.outputs.reshape(-1, 8)
    np.testing.assert_allclose(direct.concept_pool.sum(axis=-1) + direct.empty_coalition_value, outputs, atol=1e-8)
    assert not np.allclose(direct.concept_pool, base.concept_pool)
    assert direct.to_dict()["concept_estimator"] == "direct"


def test_reference_point_estimate_is_the_first_pool_member(small_problem):
    p, X_test = small_problem
    ex = Explainer(model_classes=[("gbm", GradientBoostingRegressor(n_estimators=30, random_state=0))], n_resamples=3, background_size=30, n_concepts_grid=(2,), fuzzifier_grid=(2.0,), alpha_grid=(0.0,), point_estimate="reference", feature_weights="importance", random_state=0)
    e = ex.fit(p.X, p.y).explain(X_test[:6])
    np.testing.assert_allclose(e.concept_centroids, e.concept_pool[0])
    # the feature level redistribution keeps the total credit of every instance
    np.testing.assert_allclose(e.feature_attribution().sum(axis=1), e.concept_centroids.sum(axis=1), atol=1e-9)


def test_invalid_estimator_options_are_rejected(small_problem):
    p, X_test = small_problem
    with pytest.raises(ValueError):
        concept_shapley_values(_nonlinear, X_test[:2], p.X[:20], np.eye(p.n_features), imputation="other")
    ex = Explainer(model_classes=[("gbm", GradientBoostingRegressor(n_estimators=10, random_state=0))], n_resamples=1, background_size=20, n_concepts_grid=(2,), fuzzifier_grid=(2.0,), alpha_grid=(0.0,), concept_estimator="other", random_state=0)
    with pytest.raises(ValueError, match="concept_estimator"):
        ex.fit(p.X, p.y).explain(X_test[:2])
