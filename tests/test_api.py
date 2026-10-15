"""The shap style interface of the explanation: values, base values, data, indexing, conversions and views."""

import numpy as np
import pytest
from sklearn.ensemble import GradientBoostingRegressor

import sparc


@pytest.fixture(scope="module")
def explanation(small_problem):
    p, X_test = small_problem
    ex = sparc.Explainer(model_classes=[("gbm", GradientBoostingRegressor(n_estimators=40, random_state=0))], n_resamples=4, background_size=40, n_concepts_grid=(2, 3), fuzzifier_grid=(2.0,), alpha_grid=(0.0, 0.5), point_estimate="reference", random_state=0)
    ex.fit(p.X, p.y, feature_names=[f"f{j}" for j in range(p.n_features)])
    return ex, ex(X_test[:12])


def test_call_is_explain(small_problem, explanation):
    p, X_test = small_problem
    ex, e = explanation
    e2 = ex.explain(X_test[:12])
    np.testing.assert_allclose(e.values, e2.values)
    assert len(e) == 12 and e.values.shape == (12, e.n_concepts)


def test_values_and_base_values_reproduce_the_output(explanation):
    _, e = explanation
    np.testing.assert_allclose(e.values.sum(axis=1) + e.base_values, e.output, atol=1e-6)
    assert e.base_values.shape == (12,)


def test_indexing_slices_every_per_instance_quantity(explanation):
    _, e = explanation
    one = e[3]
    assert len(one) == 1 and one.n_concepts == e.n_concepts
    np.testing.assert_allclose(one.values[0], e.values[3])
    np.testing.assert_allclose(one.quantiles[0], e.quantiles[3])
    np.testing.assert_allclose(one.sign_confidence[0], e.sign_confidence[3])
    np.testing.assert_allclose(one.concept_pool[:, 0], e.concept_pool[:, 3])
    np.testing.assert_allclose(one.output, e.output[3:4])
    assert one.instance_index.tolist() == [3]
    assert one.describe(0).startswith("Instance 0")
    sl = e[2:5]
    assert len(sl) == 3 and sl.instance_index.tolist() == [2, 3, 4]
    sel = e[[0, 5, 11]]
    np.testing.assert_allclose(sel.values, e.values[[0, 5, 11]])
    assert e[-1].instance_index.tolist() == [11]
    with pytest.raises(IndexError):
        e[12]
    # the partition and the selection are shared
    assert sel.partition is e.partition and sel.selection is e.selection
    assert sel.retained is e.retained


def test_data_are_concept_scores_from_standardized_features(explanation):
    _, e = explanation
    D = e.data
    assert D.shape == (12, e.n_concepts)
    U = e.partition.U
    Z = (e.X - e.feature_mean) / np.where(e.feature_std > 1e-12, e.feature_std, 1.0)
    np.testing.assert_allclose(D, Z @ (U / U.sum(axis=0)), atol=1e-9)
    assert e.feature_names == [f"f{j}" for j in range(U.shape[0])]
    assert e.names == e.concept_names
    np.testing.assert_allclose(e.importance, np.abs(e.values).mean(axis=0))


def test_to_shap_feature_and_concept_level(explanation):
    shap = pytest.importorskip("shap")
    _, e = explanation
    sf = e.to_shap()
    assert isinstance(sf, shap.Explanation)
    assert sf.values.shape == (12, e.partition.U.shape[0])
    assert list(sf.feature_names) == e.feature_names
    np.testing.assert_allclose(sf.values.sum(axis=1), e.values.sum(axis=1), atol=1e-9)
    np.testing.assert_allclose(sf.base_values, e.base_values)
    sc = e.to_shap(level="concept")
    assert sc.values.shape == (12, e.n_concepts) and list(sc.feature_names) == e.concept_names
    np.testing.assert_allclose(sc.data, e.data)
    with pytest.raises(ValueError):
        e.to_shap(level="other")


def test_list_dataframe_and_labels_views(explanation):
    _, e = explanation
    rows = e.as_list(0)
    assert len(rows) == e.n_concepts
    assert all(len(r) == 4 for r in rows)
    assert [abs(r[1]) for r in rows] == sorted([abs(r[1]) for r in rows], reverse=True)
    assert set(r[2] for r in rows) <= set(e.label_set.names)
    assert len(e.as_list(0, retained_only=True)) == len(e.retained)
    assert e.labels(0) == [r[2] for r in sorted(rows, key=lambda r: e.concept_names.index(r[0]))]
    df = e.as_dataframe()
    assert df.shape[0] == 12 * e.n_concepts
    assert {"instance", "concept", "value", "label", "sign_confidence", "retained"} <= set(df.columns)
    assert df["instance"].nunique() == 12
    assert e.fuzzy(0, 0).centroid() == pytest.approx(e.fuzzy_attribution(0, 0).centroid())


def test_html_card(explanation):
    _, e = explanation
    html = e.as_html(1)
    assert html.startswith("<div") and html.endswith("</div>")
    for name in e.concept_names:
        assert name in html
    assert "instance 1" in html
    short = e.as_html(1, max_display=1)
    assert short.count("<tr") == 2  # header and one row
    out = e.show_in_notebook(1)
    assert out is None or out == html
    assert "Explanation(n_explain=12" in repr(e)
