"""The SPARC explainer: a scikit learn style interface tying the modules together.

Usage
-----
>>> explainer = Explainer(n_resamples=50, random_state=0).fit(X_train, y_train)
>>> explanation = explainer.explain(X_test[:200])
>>> explanation.describe(0)
"""

from __future__ import annotations

import html as _html
import json
import warnings
from dataclasses import dataclass, field, replace
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from sparc.attribution import AttributionEngine, AttributionResult, aggregate_to_concepts, redistribute_to_features
from sparc.concepts import ConceptFormer, ConceptPartition, correlation_matrix, correlation_profile_embedding, identity_partition
from sparc.fuzzification import (
    IntervalType2FuzzyNumber,
    LinguisticLabelSet,
    TrapezoidalFuzzyNumber,
    centroid_array,
    disagreement_index,
    fuzzify_array,
    fuzzy_ranking,
    possibility_matrix,
    sign_confidence,
    standardization_scale,
    tfn_from_quantiles,
)
from sparc.selection import (
    ExplanationConfiguration,
    ParetoSelector,
    PartitionPool,
    SelectionResult,
    concept_selection_membership,
    retained_concepts,
)


@dataclass
class Explanation:
    """Outputs of SPARC for a set of explained instances (Section 3.6 of the guide)."""

    partition: ConceptPartition
    configuration: ExplanationConfiguration
    retained: np.ndarray
    activity: np.ndarray
    quantiles: np.ndarray
    concept_pool: np.ndarray
    sign_confidence: np.ndarray
    disagreement: np.ndarray
    type2_mask: np.ndarray
    global_quantiles: np.ndarray
    possibility: np.ndarray
    ranking: np.ndarray
    dominance: np.ndarray
    linguistic: np.ndarray
    label_set: LinguisticLabelSet
    scale: float
    feature_selection_membership: np.ndarray
    concept_selection_membership: np.ndarray
    selection: SelectionResult
    attribution: AttributionResult
    reference_output: np.ndarray
    disagreement_threshold: float
    fuzzy_shape: str = "trapezoidal"
    support_percentiles: tuple[float, float] = (5.0, 95.0)
    core_percentiles: tuple[float, float] = (25.0, 75.0)
    pools: dict = field(default_factory=dict, repr=False)
    concept_estimator: str = "aggregate"
    point_estimate: str = "reference"
    feature_weights: str = "membership"
    empty_coalition_value: np.ndarray | None = None
    feature_mean: np.ndarray | None = None
    feature_std: np.ndarray | None = None
    instance_index: np.ndarray | None = None
    configuration_index: int | None = None
    linguistic_scale_method: object = "mean_abs"

    # ------------------------------------------------------------------
    # array interface: values, base values, data, names, indexing
    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        return f"Explanation(n_explain={self.n_explain}, n_concepts={self.n_concepts}, retained={[self.concept_names[k] for k in self.retained]}, estimator={self.concept_estimator}, point={self.point_estimate})"

    def __len__(self) -> int:
        return self.n_explain

    def __getitem__(self, item) -> "Explanation":
        """Explanation of a subset of the explained instances: ``e[i]``, ``e[2:5]`` or ``e[[0, 3]]``.

        Every per instance quantity is sliced; the partition, the selection
        and the global quantities are shared. Integer indexing keeps one
        instance (``len(e[i]) == 1``) so that the per instance plots and
        ``describe(0)`` apply to the result.
        """
        n = self.n_explain
        if isinstance(item, (int, np.integer)):
            idx = np.array([int(item) % n if item < 0 else int(item)])
            if idx[0] >= n:
                raise IndexError(f"instance {item} out of range for {n} explained instances")
        elif isinstance(item, slice):
            idx = np.arange(n)[item]
        else:
            idx = np.asarray(item)
            if idx.dtype == bool:
                idx = np.flatnonzero(idx)
            idx = idx.astype(int)
        return self.subset(idx)

    def subset(self, indices) -> "Explanation":
        idx = np.asarray(indices, dtype=int)
        attr = self.attribution
        new_attr = replace(
            attr,
            values=attr.values[:, :, idx],
            outputs=attr.outputs[:, :, idx],
            reference_output=attr.reference_output[idx],
            X_explain=None if attr.X_explain is None else attr.X_explain[idx],
        )
        original = self.instance_index if self.instance_index is not None else np.arange(self.n_explain)
        pools = {name: PartitionPool(pp.partition, pp.pool[:, idx], pp.quantiles[idx], pp.global_quantiles) for name, pp in self.pools.items()}
        return replace(
            self,
            pools=pools,
            quantiles=self.quantiles[idx],
            concept_pool=self.concept_pool[:, idx],
            sign_confidence=self.sign_confidence[idx],
            disagreement=self.disagreement[idx],
            type2_mask=self.type2_mask[idx],
            linguistic=self.linguistic[idx],
            reference_output=self.reference_output[idx],
            attribution=new_attr,
            empty_coalition_value=None if self.empty_coalition_value is None else self.empty_coalition_value[:, idx],
            instance_index=np.asarray(original)[idx],
        )

    @property
    def values(self) -> np.ndarray:
        """Point concept attribution, shape (n_explain, K)."""
        return self.concept_centroids

    @property
    def base_values(self) -> np.ndarray:
        """Value the concept attributions are measured from, shape (n_explain,).

        For the aggregated estimator it is the expected output of the
        reference model over its background sample; for the direct estimator
        it is the value of the empty coalition of the reference model. With
        ``point_estimate='reference'`` ``values.sum(axis=1) + base_values``
        equals the reference model output of every instance.
        """
        n = self.n_explain
        if self.concept_estimator == "direct" and self.empty_coalition_value is not None:
            return np.asarray(self.empty_coalition_value[0], dtype=float)
        return np.full(n, float(self.attribution.expected[0, 0]))

    @property
    def output(self) -> np.ndarray:
        """Output of the reference model for every explained instance."""
        return np.asarray(self.reference_output, dtype=float)

    @property
    def feature_names(self) -> list[str]:
        names = list(getattr(self.partition, "feature_names", []) or [])
        d = self.partition.U.shape[0]
        return names if len(names) == d else [f"x{j}" for j in range(d)]

    @property
    def names(self) -> list[str]:
        """Concept names."""
        return self.concept_names

    @property
    def X(self) -> np.ndarray | None:
        """The explained instances (feature level), when the explanation carries them."""
        return self.attribution.X_explain

    @property
    def data(self) -> np.ndarray | None:
        """Concept scores of the explained instances, shape (n_explain, K).

        The membership weighted mean of the standardized features of every
        concept (standardized with the training mean and standard deviation),
        used to colour the beeswarm and the dependence plots.
        """
        X = self.X
        if X is None:
            return None
        mean = self.feature_mean if self.feature_mean is not None else X.mean(axis=0)
        std = self.feature_std if self.feature_std is not None else X.std(axis=0)
        std = np.where(np.asarray(std) > 1e-12, std, 1.0)
        Z = (X - mean) / std
        U = self.partition.U
        mass = U.sum(axis=0)
        W = U / np.where(mass <= 0, 1.0, mass)[None, :]
        return Z @ W

    @property
    def importance(self) -> np.ndarray:
        """Mean absolute point attribution of every concept over the explained instances."""
        return np.abs(self.values).mean(axis=0)

    def fuzzy(self, instance: int, concept: int):
        """Alias of :meth:`fuzzy_attribution`."""
        return self.fuzzy_attribution(instance, concept)

    def labels(self, instance: int) -> list[str]:
        """Best linguistic label of every concept for one instance."""
        idx, _ = self.best_labels()
        return [self.label_set.names[int(k)] for k in idx[instance]]

    def as_list(self, instance: int = 0, retained_only: bool = False) -> list[tuple]:
        """Concepts of one instance as (name, value, label, sign confidence), most important first."""
        idx, deg = self.best_labels()
        v = self.values[instance]
        retained = set(self.retained.tolist())
        order = np.argsort(-np.abs(v))
        out = []
        for k in order:
            if retained_only and k not in retained:
                continue
            out.append((self.concept_names[k], float(v[k]), self.label_set.names[int(idx[instance, k])], float(self.sign_confidence[instance, k])))
        return out

    def as_dataframe(self) -> pd.DataFrame:
        """Long table with one row per explained instance and concept."""
        idx, deg = self.best_labels()
        v = self.values
        retained = set(self.retained.tolist())
        rows = []
        inst = self.instance_index if self.instance_index is not None else np.arange(self.n_explain)
        for i in range(self.n_explain):
            for k in range(self.n_concepts):
                q = self.quantiles[i, k]
                rows.append(
                    {
                        "instance": int(inst[i]),
                        "concept": self.concept_names[k],
                        "value": float(v[i, k]),
                        "median": float(q[2]),
                        "support_low": float(q[0]),
                        "core_low": float(q[1]),
                        "core_high": float(q[3]),
                        "support_high": float(q[4]),
                        "sign_confidence": float(self.sign_confidence[i, k]),
                        "disagreement": float(self.disagreement[i, k]),
                        "type2": bool(self.type2_mask[i, k]),
                        "label": self.label_set.names[int(idx[i, k])],
                        "label_degree": float(deg[i, k]),
                        "retained": k in retained,
                        "selection_membership": float(self.concept_selection_membership[k]),
                    }
                )
        return pd.DataFrame(rows)

    def as_html(self, instance: int = 0, retained_only: bool = False, max_display: int | None = None) -> str:
        """Self contained HTML card of one instance: output, concept bars with labels, support and sign confidence."""
        rows = self.as_list(instance, retained_only=retained_only)
        if max_display is not None:
            rows = rows[:max_display]
        vmax = max([abs(r[1]) for r in rows] + [1e-12])
        inst = int(self.instance_index[instance]) if self.instance_index is not None else instance
        idx, _ = self.best_labels()
        names = {n: k for k, n in enumerate(self.concept_names)}
        retained = set(self.retained.tolist())
        parts = [
            '<div style="font-family:Georgia,serif;max-width:720px;border:1px solid #e6e5e1;border-radius:6px;padding:12px 16px;color:#0b0b0b">',
            f'<div style="font-size:15px;margin-bottom:6px"><b>SPARC explanation</b> of instance {inst}: model output {self.output[instance]:.4g}, base value {self.base_values[instance]:.4g}</div>',
            '<table style="border-collapse:collapse;width:100%;font-size:13px">',
            '<tr style="color:#52514e;text-align:left"><th style="padding:4px 6px">concept</th><th style="padding:4px 6px">attribution</th><th style="padding:4px 6px">label</th><th style="padding:4px 6px">support</th><th style="padding:4px 6px">sign conf.</th></tr>',
        ]
        for name, value, label, sc in rows:
            k = names[name]
            q = self.quantiles[instance, k]
            width = 100.0 * abs(value) / vmax
            colour = "#2a78d6" if value >= 0 else "#eb6834"
            weight = "600" if k in retained else "400"
            bar = f'<div style="display:flex;align-items:center;gap:6px"><div style="height:10px;width:{width:.0f}%;max-width:160px;background:{colour};border-radius:2px"></div><span>{value:+.3g}</span></div>'
            note = " (type 2)" if self.type2_mask[instance, k] else ""
            parts.append(
                f'<tr style="border-top:1px solid #e6e5e1;font-weight:{weight}"><td style="padding:4px 6px">{_html.escape(name)}</td><td style="padding:4px 6px;min-width:200px">{bar}</td>'
                f'<td style="padding:4px 6px">{_html.escape(label)}{note}</td><td style="padding:4px 6px;color:#52514e">[{q[0]:+.3g}, {q[4]:+.3g}]</td><td style="padding:4px 6px">{sc:.2f}</td></tr>'
            )
        cfg = self.configuration
        parts.append("</table>")
        parts.append(f'<div style="font-size:12px;color:#52514e;margin-top:6px">{cfg.n_concepts} concepts, {len(self.retained)} retained at alpha {cfg.alpha:.2f}; bold rows are retained by the knee explanation; blue positive, orange negative.</div></div>')
        return "".join(parts)

    def show_in_notebook(self, instance: int = 0, **kwargs):
        """Display the HTML card in a Jupyter notebook (returns the HTML string outside a notebook)."""
        html = self.as_html(instance, **kwargs)
        try:
            from IPython.display import HTML, display

            display(HTML(html))
        except ImportError:
            return html

    # ------------------------------------------------------------------
    @property
    def n_explain(self) -> int:
        return int(self.quantiles.shape[0])

    @property
    def n_concepts(self) -> int:
        return int(self.quantiles.shape[1])

    @property
    def concept_names(self) -> list[str]:
        return self.partition.concept_names()

    @property
    def membership(self) -> np.ndarray:
        return self.partition.U

    @property
    def concept_centroids(self) -> np.ndarray:
        """Point concept attribution, shape (n_explain, K).

        The centroid of the trapezoidal fuzzy number by default; with
        ``point_estimate='reference'`` the attribution of the reference model
        on the original data (pool member b = 0, m = 0), the fuzzy number and
        the sign confidence still describing the whole pool.
        """
        if self.point_estimate == "reference":
            return np.asarray(self.concept_pool[0], dtype=float)
        return centroid_array(self.quantiles)

    @property
    def concept_medians(self) -> np.ndarray:
        return self.quantiles[..., 2]

    def feature_attribution(self, retained_only: bool = False) -> np.ndarray:
        """Defuzzified feature level vector obtained by redistributing concept centroids.

        With ``feature_weights='importance'`` the share of a feature in its
        concept follows its mean absolute feature level attribution over the
        pool instead of its membership alone (the concept totals are
        unchanged).
        """
        C = self.concept_centroids.copy()
        if retained_only:
            mask = np.zeros(self.n_concepts, dtype=bool)
            mask[self.retained] = True
            C[:, ~mask] = 0.0
        weights = None
        if self.feature_weights == "importance":
            weights = np.abs(self.attribution.values).mean(axis=(0, 1, 2))
        return redistribute_to_features(C, self.partition.U, weights)

    def fuzzy_attribution(self, instance: int, concept: int):
        """Fuzzy attribution of one instance and concept; type 2 when the model classes disagree."""
        if self.type2_mask[instance, concept]:
            return self.type2_number(instance, concept)
        return tfn_from_quantiles(self.quantiles[instance, concept])

    def type2_number(self, instance: int, concept: int) -> IntervalType2FuzzyNumber:
        B, M = self.attribution.n_resamples, self.attribution.n_model_classes
        pool = self.concept_pool.reshape(B, M, self.n_explain, self.n_concepts)
        members = []
        for m in range(M):
            q = fuzzify_array(pool[:, m, instance, concept], axis=0, shape=self.fuzzy_shape, support_percentiles=self.support_percentiles, core_percentiles=self.core_percentiles)
            members.append(tfn_from_quantiles(q))
        return IntervalType2FuzzyNumber(members)

    def global_fuzzy_importance(self) -> list[TrapezoidalFuzzyNumber]:
        return [tfn_from_quantiles(q) for q in self.global_quantiles]

    def best_labels(self) -> tuple[np.ndarray, np.ndarray]:
        """Most compatible label index and degree for every instance and concept.

        Ties at the maximal compatibility are broken by the tie breaking rule
        of :meth:`LinguisticLabelSet.best_label` (the label whose center is
        closest to the centroid of the standardized attribution).
        """
        return self.label_set.best_label_array(self.quantiles, self.scale, profile=self.linguistic)

    def instance_frame(self, instance: int) -> pd.DataFrame:
        """Table with one row per concept for an explained instance."""
        names = self.concept_names
        idx, deg = self.best_labels()
        rows = []
        for k in range(self.n_concepts):
            F = tfn_from_quantiles(self.quantiles[instance, k])
            rows.append(
                {
                    "concept": names[k],
                    "retained": bool(k in set(self.retained.tolist())),
                    "centroid": F.centroid(),
                    "median": float(self.quantiles[instance, k, 2]),
                    "support_low": F.a,
                    "core_low": F.b,
                    "core_high": F.c,
                    "support_high": F.d,
                    "sign_confidence": float(self.sign_confidence[instance, k]),
                    "disagreement": float(self.disagreement[instance, k]),
                    "type2": bool(self.type2_mask[instance, k]),
                    "label": self.label_set.names[int(idx[instance, k])],
                    "label_degree": float(deg[instance, k]),
                    "selection_membership": float(self.concept_selection_membership[k]),
                }
            )
        df = pd.DataFrame(rows)
        return df.reindex(df["centroid"].abs().sort_values(ascending=False).index).reset_index(drop=True)

    def global_frame(self) -> pd.DataFrame:
        names = self.concept_names
        G = self.global_fuzzy_importance()
        rank_pos = np.empty(self.n_concepts, dtype=int)
        rank_pos[self.ranking] = np.arange(1, self.n_concepts + 1)
        rows = []
        for k in range(self.n_concepts):
            rows.append(
                {
                    "concept": names[k],
                    "rank": int(rank_pos[k]),
                    "retained": bool(k in set(self.retained.tolist())),
                    "importance_centroid": G[k].centroid(),
                    "importance_support_low": G[k].a,
                    "importance_support_high": G[k].d,
                    "dominance": float(self.dominance[k]),
                    "activity": float(self.activity[k]),
                    "fuzziness": float(self.partition.fuzziness[k]),
                    "selection_membership": float(self.concept_selection_membership[k]),
                }
            )
        return pd.DataFrame(rows).sort_values("rank").reset_index(drop=True)

    def describe(self, instance: int, retained_only: bool = True) -> str:
        """Linguistic explanation of one instance."""
        lines = [f"Instance {instance}: model output {self.reference_output[instance]:.4g}"]
        cfg = self.configuration
        lines.append(f"Configuration: K={cfg.n_concepts}, fuzzifier={cfg.fuzzifier}, sparsity={cfg.sparsity}, alpha={cfg.alpha:.2f}")
        idx, deg = self.best_labels()
        order = np.argsort(-np.abs(self.concept_centroids[instance]))
        names = self.concept_names
        retained = set(self.retained.tolist())
        for k in order:
            if retained_only and k not in retained:
                continue
            F = self.fuzzy_attribution(instance, k)
            if isinstance(F, IntervalType2FuzzyNumber):
                lo, hi = F.centroid_interval()
                prof = self.label_set.profile_type2(F, self.scale)
                label = max(prof, key=lambda n: prof[n][1])
                lines.append(
                    f"  {names[k]}: {label} (compatibility {prof[label][0]:.2f} to {prof[label][1]:.2f}, model classes disagree, "
                    f"disagreement {self.disagreement[instance, k]:.2f}), centroid between {lo:+.3g} and {hi:+.3g}, "
                    f"sign confidence {self.sign_confidence[instance, k]:.2f}, selection membership {self.concept_selection_membership[k]:.2f}"
                )
            else:
                lines.append(
                    f"  {names[k]}: {self.label_set.names[int(idx[instance, k])]} (compatibility {deg[instance, k]:.2f}), "
                    f"centroid {F.centroid():+.3g}, support [{F.a:+.3g}, {F.d:+.3g}], core [{F.b:+.3g}, {F.c:+.3g}], "
                    f"sign confidence {self.sign_confidence[instance, k]:.2f}, disagreement {self.disagreement[instance, k]:.2f}, "
                    f"selection membership {self.concept_selection_membership[k]:.2f}"
                )
        return "\n".join(lines)

    @property
    def front(self) -> pd.DataFrame:
        """The Pareto optimal configurations with their objectives and flags (the front as a table)."""
        return self.selection.front

    @property
    def configurations(self) -> pd.DataFrame:
        """Every evaluated configuration with its objectives, ``pareto`` and ``knee`` flags."""
        return self.selection.table

    @property
    def fidelity_optimal(self) -> int:
        """Index of the front member with the smallest fidelity loss (ties broken by complexity, then instability)."""
        F = np.asarray(self.selection.objectives, dtype=float)
        mask = np.asarray(self.selection.pareto, dtype=bool)
        idx = np.flatnonzero(mask)
        order = np.lexsort((F[idx, 2], F[idx, 1], F[idx, 0]))
        return int(idx[order[0]])

    def front_members(self, distinct: bool = True) -> list[int]:
        """Indices of the front members ordered from the simplest to the most complex explanation."""
        table = self.selection.distinct_front if distinct else self.selection.front
        return [int(i) for i in table.sort_values(["complexity", "fidelity_loss"]).index]

    def at(self, index: int) -> "Explanation":
        """The explanation of another evaluated configuration, for example another member of the front.

        The pools of the candidate partitions are reused, so the call is
        cheap; for the direct estimator only the partition of the selected
        configuration carries direct concept values, the other partitions
        keep the aggregated ones. Instance subsets are preserved.
        """
        e = build_explanation(
            self.selection,
            int(index),
            self.pools,
            self.attribution,
            disagreement_threshold=self.disagreement_threshold,
            label_set=self.label_set,
            linguistic_scale=self.linguistic_scale_method,
            fuzzy_shape=self.fuzzy_shape,
            support_percentiles=self.support_percentiles,
            core_percentiles=self.core_percentiles,
            concept_estimator=self.concept_estimator,
            point_estimate=self.point_estimate,
            feature_weights=self.feature_weights,
            empty_coalition_value=self.empty_coalition_value if self.selection.configurations[int(index)].partition == self.configuration.partition else None,
            feature_mean=self.feature_mean,
            feature_std=self.feature_std,
        )
        if self.instance_index is not None:
            e = replace(e, instance_index=self.instance_index)
        return e

    def to_dict(self) -> dict:
        """JSON serializable summary (without the raw pool)."""
        return {
            "configuration": self.configuration.as_dict(),
            "retained": [int(k) for k in self.retained],
            "concept_names": self.concept_names,
            "membership": self.partition.U.tolist(),
            "quantiles": self.quantiles.tolist(),
            "sign_confidence": self.sign_confidence.tolist(),
            "disagreement": self.disagreement.tolist(),
            "type2": self.type2_mask.tolist(),
            "global_quantiles": self.global_quantiles.tolist(),
            "possibility": self.possibility.tolist(),
            "ranking": [int(k) for k in self.ranking],
            "labels": self.label_set.names,
            "linguistic": self.linguistic.tolist(),
            "scale": self.scale,
            "feature_selection_membership": self.feature_selection_membership.tolist(),
            "concept_selection_membership": self.concept_selection_membership.tolist(),
            "front": json.loads(self.selection.front.to_json(orient="records")),
            "classes": None if self.attribution.classes is None else np.asarray(self.attribution.classes).tolist(),
            "explained_class": self.attribution.explained_class,
            "concept_estimator": self.concept_estimator,
            "point_estimate": self.point_estimate,
            "feature_weights": self.feature_weights,
        }

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf8") as fh:
            json.dump(self.to_dict(), fh)


class Explainer(BaseEstimator):
    """Pareto optimal fuzzy concept attribution.

    Parameters
    ----------
    model_classes : list of estimators or (name, estimator) pairs, optional
        Model classes used to quantify model dependence. The first is the
        reference model class. None uses gradient boosting, random forest and a
        multilayer perceptron.
    n_resamples : int
        Number of bootstrap resamples B (resample 0 is the original data).
    background_size : int or None
        Background sample size for the Shapley computation.
    explainer : {'auto', 'tree', 'linear', 'kernel', 'exact', 'permutation'}
    n_concepts_grid : iterable of int
        Candidate numbers of concepts K. A value equal to the number of
        features d gives the identity partition (one crisp concept per
        feature, the K = d case of the reduction to feature level Shapley values), which is also the
        fallback when no other data driven candidate can be formed; values
        outside 1 to d are ignored with a warning.
    fuzzifier_grid : iterable of float
    concept_distance : {'correlation', 'loading'}
    correlation : {'pearson', 'spearman'}
    apriori_groups : dict or list, optional
        Known grouping of the features, included as a candidate partition.
    partition_source : {'data', 'apriori', 'both'}
        Which candidate partitions enter the selection.
    consensus : bool
        Consensus fuzzy c means over resamples of the instances.
    crisp_concepts : bool
        Ablation: harden the concept partitions.
    fuzzy_shape : {'trapezoidal', 'triangular'}
    support_percentiles, core_percentiles : pairs of floats
    disagreement_threshold : float
        Disagreement index above which an attribution is upgraded to type 2.
    linguistic_labels : LinguisticLabelSet, optional
    linguistic_scale : {'mean_abs', 'output_sd'} or float
    alpha_grid : iterable of float
    sparsity_grid : iterable of int or None
    solver : {'grid', 'nsga2', 'fixed'}
        'fixed' evaluates only ``fixed_configuration`` (ablation).
    fixed_configuration : dict, optional
        Keys n_concepts, fuzzifier, sparsity, alpha (or partition='apriori').
    knee_method : {'utopia', 'hyperplane'}
    surrogate : {'linear', 'tree'}
    concept_estimator : {'aggregate', 'direct'}
        How the concept attributions of the selected partition are estimated.
        'aggregate' forms them from the feature level Shapley values of the
        pool (Equation 1), which is also what the Pareto selection uses for
        every candidate partition. 'direct' recomputes the pool of the
        selected partition with Shapley values over the fuzzy concepts
        themselves (:mod:`sparc.concept_shapley`), whose coalitions remove
        features to the degree of their membership exactly as the deletion
        and insertion tests do.
    direct_imputation : {'conditional', 'marginal'}
        Imputation of the removed features in the direct estimator.
    point_estimate : {'reference', 'centroid'}
        Point attribution reported beside the fuzzy number: the attribution
        of the reference model on the original data (default; the fuzzy
        number, the sign confidence and the disagreement still describe the
        whole pool), or the centroid of the trapezoid over the pool.
    feature_weights : {'membership', 'importance'}
        How a concept's credit is redistributed to its features by
        :meth:`Explanation.feature_attribution`.
    task : {'auto', 'regression', 'classification'}
        See :class:`sparc.attribution.AttributionEngine`.
    target_class : int or label
        Class whose probability is explained for classifiers, given as one of
        the class labels or as an index into the sorted classes; the resolved
        class is exposed as ``classes_`` and ``target_class_`` after fitting.
    n_jobs : int
    random_state : int or None
    """

    def __init__(
        self,
        model_classes=None,
        n_resamples: int = 50,
        background_size: int | None = 100,
        explainer: str = "auto",
        n_concepts_grid: Sequence[int] = (2, 3, 4, 5, 6),
        fuzzifier_grid: Sequence[float] = (1.5, 2.0, 2.5),
        concept_distance: str = "correlation",
        correlation: str = "pearson",
        apriori_groups=None,
        partition_source: str = "both",
        consensus: bool = False,
        crisp_concepts: bool = False,
        fuzzy_shape: str = "trapezoidal",
        support_percentiles: tuple[float, float] = (5.0, 95.0),
        core_percentiles: tuple[float, float] = (25.0, 75.0),
        disagreement_threshold: float = 0.5,
        linguistic_labels: LinguisticLabelSet | None = None,
        linguistic_scale="mean_abs",
        alpha_grid: Sequence[float] = (0.0, 0.25, 0.5, 0.75, 1.0),
        sparsity_grid: Sequence[int] | None = None,
        solver: str = "grid",
        fixed_configuration: dict | None = None,
        knee_method: str = "utopia",
        surrogate: str = "linear",
        cv: int = 5,
        nsga_pop_size: int = 40,
        nsga_generations: int = 30,
        concept_estimator: str = "aggregate",
        direct_imputation: str = "conditional",
        point_estimate: str = "reference",
        feature_weights: str = "membership",
        task: str = "auto",
        target_class: int = 1,
        kernel_nsamples="auto",
        permutation_max_evals="auto",
        n_jobs: int = 1,
        random_state: int | None = None,
        verbose: int = 0,
    ):
        self.model_classes = model_classes
        self.concept_estimator = concept_estimator
        self.direct_imputation = direct_imputation
        self.point_estimate = point_estimate
        self.feature_weights = feature_weights
        self.n_resamples = n_resamples
        self.background_size = background_size
        self.explainer = explainer
        self.n_concepts_grid = n_concepts_grid
        self.fuzzifier_grid = fuzzifier_grid
        self.concept_distance = concept_distance
        self.correlation = correlation
        self.apriori_groups = apriori_groups
        self.partition_source = partition_source
        self.consensus = consensus
        self.crisp_concepts = crisp_concepts
        self.fuzzy_shape = fuzzy_shape
        self.support_percentiles = support_percentiles
        self.core_percentiles = core_percentiles
        self.disagreement_threshold = disagreement_threshold
        self.linguistic_labels = linguistic_labels
        self.linguistic_scale = linguistic_scale
        self.alpha_grid = alpha_grid
        self.sparsity_grid = sparsity_grid
        self.solver = solver
        self.fixed_configuration = fixed_configuration
        self.knee_method = knee_method
        self.surrogate = surrogate
        self.cv = cv
        self.nsga_pop_size = nsga_pop_size
        self.nsga_generations = nsga_generations
        self.task = task
        self.target_class = target_class
        self.kernel_nsamples = kernel_nsamples
        self.permutation_max_evals = permutation_max_evals
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose

    # ------------------------------------------------------------------
    def fit(self, X, y, feature_names: Sequence[str] | None = None):
        """Fit the pool of models and form the candidate concept partitions."""
        if feature_names is None and hasattr(X, "columns"):
            feature_names = [str(c) for c in X.columns]
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.feature_names_ = list(feature_names) if feature_names is not None else [f"x{j}" for j in range(X.shape[1])]
        self.engine_ = AttributionEngine(
            model_classes=self.model_classes,
            n_resamples=self.n_resamples,
            background_size=self.background_size,
            explainer=self.explainer,
            task=self.task,
            target_class=self.target_class,
            kernel_nsamples=self.kernel_nsamples,
            permutation_max_evals=self.permutation_max_evals,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
            verbose=self.verbose,
        ).fit(X, y)
        self.task_ = self.engine_.task_
        self._fit_partitions(X)
        return self

    @property
    def reference_model_(self):
        return self.engine_.reference_model_

    @property
    def classes_(self):
        """Sorted class labels of the fitted engine (classification only)."""
        return self.engine_.classes_

    @property
    def target_class_(self):
        """Index into ``classes_`` of the explained class; None for regression."""
        return self.engine_.target_class_

    # ------------------------------------------------------------------
    def _partition_pool(self, partition: ConceptPartition, attr: AttributionResult) -> PartitionPool:
        concept = aggregate_to_concepts(attr.values, partition.U)
        B, M, n, K = concept.shape
        pool = concept.reshape(B * M, n, K)
        quantiles = fuzzify_array(pool, axis=0, shape=self.fuzzy_shape, support_percentiles=self.support_percentiles, core_percentiles=self.core_percentiles)
        global_imp = np.abs(pool).mean(axis=1)
        global_quantiles = fuzzify_array(global_imp, axis=0, shape=self.fuzzy_shape, support_percentiles=self.support_percentiles, core_percentiles=self.core_percentiles)
        return PartitionPool(partition, pool, quantiles, global_quantiles)

    @classmethod
    def from_engine(cls, engine: AttributionEngine, feature_names: Sequence[str] | None = None, **kwargs) -> "Explainer":
        """Build an explainer around an already fitted attribution engine (shares the pool of models).

        The engine level parameters (model_classes, n_resamples,
        background_size, explainer, task, target_class, kernel_nsamples,
        permutation_max_evals, n_jobs, random_state and verbose) are copied
        from the engine so that ``get_params`` describes the pool that was
        fitted. Any constructor argument, including these, may be given as a
        keyword argument and then takes precedence; a random_state override
        affects the concept formation and the Pareto selection only, since
        the pool is not refitted.
        """
        init_kwargs = dict(
            model_classes=engine.model_classes,
            n_resamples=engine.n_resamples,
            background_size=engine.background_size,
            explainer=engine.explainer,
            task=engine.task,
            target_class=engine.target_class,
            kernel_nsamples=engine.kernel_nsamples,
            permutation_max_evals=engine.permutation_max_evals,
            n_jobs=engine.n_jobs,
            random_state=engine.random_state,
            verbose=engine.verbose,
        )
        init_kwargs.update(kwargs)
        obj = cls(**init_kwargs)
        X = engine.X_train_
        obj.engine_ = engine
        obj.task_ = engine.task_
        obj.feature_names_ = list(feature_names) if feature_names is not None else [f"x{j}" for j in range(X.shape[1])]
        obj._fit_partitions(X)
        return obj

    def _candidate_partition(self, n_concepts: int, fuzzifier: float | None) -> ConceptPartition:
        """Candidate partition with K concepts; K equal to the number of features is the identity partition."""
        K = int(n_concepts)
        if K == self.n_features_in_:
            return identity_partition(K, self.feature_names_)
        return self.former_.partition(K, fuzzifier)

    def _fit_partitions(self, X: np.ndarray) -> None:
        d = X.shape[1]
        names = self.feature_names_
        if self.partition_source == "apriori" and self.apriori_groups is None:
            raise ValueError("partition_source='apriori' requires apriori_groups.")
        data_driven = self.partition_source != "apriori"
        use_apriori = self.apriori_groups is not None and self.partition_source in ("apriori", "both")
        requested = [int(k) for k in self.n_concepts_grid] if data_driven else []
        grid = [k for k in requested if 1 <= k < d]
        dropped = sorted({k for k in requested if k < 1 or k > d})
        if dropped:
            warnings.warn(f"n_concepts_grid values {dropped} lie outside 1 to {d} (the number of features) and are ignored.", UserWarning, stacklevel=3)
        # K = d is the identity partition; it is also the fallback when no K is admissible for fuzzy c means
        identity_requested = data_driven and (d in requested or not grid)
        self.former_ = ConceptFormer(
            n_concepts_grid=grid,
            fuzzifier_grid=self.fuzzifier_grid,
            distance=self.concept_distance,
            correlation=self.correlation,
            apriori_groups=self.apriori_groups if use_apriori else None,
            consensus=self.consensus,
            crisp=self.crisp_concepts,
            random_state=self.random_state,
        )
        if d > 1 and (grid or use_apriori):
            self.former_.fit(X, names)
        else:
            # ConceptFormer.fit raises when it can form no candidate, which is the
            # case for an empty grid without a priori groups; the former is
            # initialized here without clustering
            self.former_.X_ = X
            self.former_.n_features_ = d
            self.former_.feature_names_ = list(names)
            self.former_._cache = {}
            self.former_.candidates_ = []
            if d > 1:
                self.former_.embedding_ = correlation_profile_embedding(X, self.correlation)
                self.former_.correlation_ = correlation_matrix(X, self.correlation)
            if use_apriori:
                self.former_.candidates_.append(self.former_.apriori_partition())
        if identity_requested:
            self.former_.candidates_.append(identity_partition(d, names))
        self.X_train_ = X
        self.n_features_in_ = d

    def __call__(self, X_explain) -> Explanation:
        """Explain the given instances; ``explainer(X)`` is ``explainer.explain(X)``."""
        return self.explain(X_explain)

    def explain(self, X_explain) -> Explanation:
        """Compute the fuzzy concept explanation of the given instances."""
        X_explain = np.asarray(X_explain, dtype=float)
        attr = self.engine_.explain(X_explain)
        return self.explain_with(attr)

    def explain_with(self, attr: AttributionResult) -> Explanation:
        """Build the explanation from a precomputed pool of feature level attributions."""
        output = attr.reference_output
        selector = ParetoSelector(
            alpha_grid=self.alpha_grid,
            sparsity_grid=self.sparsity_grid,
            solver=self.solver,
            knee_method=self.knee_method,
            surrogate=self.surrogate,
            cv=self.cv,
            nsga_pop_size=self.nsga_pop_size,
            nsga_generations=self.nsga_generations,
            random_state=self.random_state if self.random_state is not None else 0,
        )
        pools: dict[str, PartitionPool] = {}

        def get_pool(partition: ConceptPartition) -> PartitionPool:
            if partition.name not in pools:
                pools[partition.name] = self._partition_pool(partition, attr)
            return pools[partition.name]

        if self.solver == "grid":
            for part in self.former_.candidates_:
                get_pool(part)
            selection = selector.run_grid(pools, output)
        elif self.solver == "fixed":
            if not self.fixed_configuration:
                raise ValueError("solver='fixed' requires fixed_configuration.")
            fc = dict(self.fixed_configuration)
            if fc.get("partition") == "apriori":
                part = self.former_.apriori_partition()
            else:
                part = self._candidate_partition(int(fc["n_concepts"]), fc.get("fuzzifier", 2.0))
            pp = get_pool(part)
            selector.alpha_grid = (float(fc.get("alpha", 0.0)),)
            selector.sparsity_grid = (int(fc.get("sparsity", part.n_concepts)),)
            selection = selector.run_grid({part.name: pp}, output)
        elif self.solver == "nsga2":
            # the K values actually fitted by fuzzy c means bound the search; the
            # a priori and identity partitions enter NSGA II as extra sources
            fcm_grid = [int(k) for k in self.former_.n_concepts_grid]
            extra = ()
            if self.apriori_groups is not None and self.partition_source in ("apriori", "both"):
                extra += ("apriori",)
            if any(p.method == "identity" for p in self.former_.candidates_):
                extra += ("identity",)

            def provider(K, m, source):
                if source == "apriori":
                    return get_pool(self.former_.apriori_partition())
                if source == "identity":
                    return get_pool(identity_partition(self.n_features_in_, self.feature_names_))
                return get_pool(self._candidate_partition(int(K), None if self.concept_distance == "loading" else float(m)))

            if self.partition_source == "apriori":
                selection = selector.run_grid({"apriori": get_pool(self.former_.apriori_partition())}, output)
            elif not fcm_grid:
                # nothing for NSGA II to search over: evaluate the candidates exhaustively
                for part in self.former_.candidates_:
                    get_pool(part)
                selection = selector.run_grid(pools, output)
            else:
                selection = selector.run_nsga2(provider, (min(fcm_grid), max(fcm_grid)), tuple(float(m) for m in self.fuzzifier_grid), output, extra)
        else:
            raise ValueError("solver must be 'grid', 'nsga2' or 'fixed'")
        empty = None
        if self.concept_estimator == "direct":
            pools, empty = self._direct_pool(selection, selection.knee, pools, attr)
        elif self.concept_estimator != "aggregate":
            raise ValueError("concept_estimator must be 'aggregate' or 'direct'")
        return self._build_explanation(selection, selection.knee, pools, attr, empty)

    def _direct_pool(self, selection: SelectionResult, index: int, pools: dict, attr: AttributionResult) -> tuple[dict, np.ndarray]:
        """Replace the pool of the selected partition by direct concept Shapley values of every pool member."""
        if attr.X_explain is None:
            raise ValueError("the direct concept estimator needs the explained instances; explain with an AttributionResult that carries X_explain")
        cfg = selection.configurations[index]
        partition = pools[cfg.partition].partition
        model_classes = None
        if attr.n_model_classes != len(self.engine_.model_classes_):
            model_classes = list(range(attr.n_model_classes))  # single model ablation: the first classes only
        pool_bm, empty = self.engine_.explain_concepts(attr.X_explain, partition.U, imputation=self.direct_imputation, model_classes=model_classes)
        B, M, n, K = pool_bm.shape
        pool = pool_bm.reshape(B * M, n, K)
        quantiles = fuzzify_array(pool, axis=0, shape=self.fuzzy_shape, support_percentiles=self.support_percentiles, core_percentiles=self.core_percentiles)
        global_imp = np.abs(pool).mean(axis=1)
        global_quantiles = fuzzify_array(global_imp, axis=0, shape=self.fuzzy_shape, support_percentiles=self.support_percentiles, core_percentiles=self.core_percentiles)
        new_pools = dict(pools)
        new_pools[cfg.partition] = PartitionPool(partition, pool, quantiles, global_quantiles)
        return new_pools, empty.reshape(B * M, n)

    def _build_explanation(self, selection: SelectionResult, index: int, pools: dict, attr: AttributionResult, empty_coalition_value: np.ndarray | None = None) -> Explanation:
        return build_explanation(
            selection,
            index,
            pools,
            attr,
            disagreement_threshold=self.disagreement_threshold,
            label_set=self.linguistic_labels if self.linguistic_labels is not None else LinguisticLabelSet(),
            linguistic_scale=self.linguistic_scale,
            fuzzy_shape=self.fuzzy_shape,
            support_percentiles=tuple(self.support_percentiles),
            core_percentiles=tuple(self.core_percentiles),
            concept_estimator=self.concept_estimator,
            point_estimate=self.point_estimate,
            feature_weights=self.feature_weights,
            empty_coalition_value=empty_coalition_value,
            feature_mean=self.engine_.X_train_.mean(axis=0),
            feature_std=self.engine_.X_train_.std(axis=0),
        )

    def explanation_at(self, explanation: Explanation, index: int) -> Explanation:
        """Rebuild the explanation for another configuration of the evaluated table (for example another front member)."""
        return self._build_explanation(explanation.selection, int(index), explanation.pools, explanation.attribution)

    def fit_explain(self, X, y, X_explain, feature_names: Sequence[str] | None = None) -> Explanation:
        return self.fit(X, y, feature_names).explain(X_explain)


def build_explanation(
    selection: SelectionResult,
    index: int,
    pools: dict,
    attr: AttributionResult,
    disagreement_threshold: float = 0.5,
    label_set: LinguisticLabelSet | None = None,
    linguistic_scale="mean_abs",
    fuzzy_shape: str = "trapezoidal",
    support_percentiles=(5.0, 95.0),
    core_percentiles=(25.0, 75.0),
    concept_estimator: str = "aggregate",
    point_estimate: str = "reference",
    feature_weights: str = "membership",
    empty_coalition_value: np.ndarray | None = None,
    feature_mean: np.ndarray | None = None,
    feature_std: np.ndarray | None = None,
) -> Explanation:
    """Assemble the explanation of configuration ``index`` from the selection and the partition pools."""
    cfg = selection.configurations[index]
    pp = pools[cfg.partition]
    partition = pp.partition
    B, M, n, d = attr.values.shape
    K = partition.n_concepts
    concept = pp.pool.reshape(B, M, n, K)
    retained, activity, _ = retained_concepts(pp.quantiles, pp.global_quantiles, cfg.alpha, cfg.sparsity)
    sc = sign_confidence(pp.pool, axis=0)
    dis = disagreement_index(concept)
    type2 = dis > disagreement_threshold
    label_set = label_set if label_set is not None else LinguisticLabelSet()
    scale = standardization_scale(concept[0, 0], linguistic_scale, attr.reference_output)
    linguistic = label_set.profile_array(pp.quantiles, scale)
    G = [tfn_from_quantiles(q) for q in pp.global_quantiles]
    poss = possibility_matrix(G)
    ranking, dominance = fuzzy_ranking(G)
    try:
        csm = selection.concept_membership(cfg.partition)
    except (KeyError, AttributeError):
        csm = concept_selection_membership(partition.U, selection.feature_selection_membership)
    return Explanation(
        partition=partition,
        configuration=cfg,
        retained=retained,
        activity=activity,
        quantiles=pp.quantiles,
        concept_pool=pp.pool,
        sign_confidence=sc,
        disagreement=dis,
        type2_mask=type2,
        global_quantiles=pp.global_quantiles,
        possibility=poss,
        ranking=ranking,
        dominance=dominance,
        linguistic=linguistic,
        label_set=label_set,
        scale=scale,
        feature_selection_membership=selection.feature_selection_membership,
        concept_selection_membership=csm,
        selection=selection,
        attribution=attr,
        reference_output=attr.reference_output,
        disagreement_threshold=disagreement_threshold,
        fuzzy_shape=fuzzy_shape,
        support_percentiles=tuple(support_percentiles),
        core_percentiles=tuple(core_percentiles),
        pools=pools,
        concept_estimator=concept_estimator,
        point_estimate=point_estimate,
        feature_weights=feature_weights,
        empty_coalition_value=empty_coalition_value,
        feature_mean=feature_mean,
        feature_std=feature_std,
        configuration_index=int(index),
        linguistic_scale_method=linguistic_scale,
    )
