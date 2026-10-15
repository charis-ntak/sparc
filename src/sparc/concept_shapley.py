"""Shapley values computed directly over fuzzy concepts (the direct concept estimator).

The aggregated estimator of Equation 1 forms a concept attribution as the
membership weighted sum of feature level Shapley values. The direct estimator
of this module treats every fuzzy concept as a player of a cooperative game.
A coalition S of concepts keeps feature j to the degree

    w_j(S) = min(1, sum_{k in S} u_jk),

the membership mass of j in S, so that the grand coalition keeps every feature
entirely whenever the rows of the membership matrix sum to one, and the empty
coalition removes every feature. The value of a coalition is the model output
of the explained instance with the remaining features removed to the degree
1 - w_j(S), by conditional imputation (mean of the nearest background rows on
the kept features, the operation that the deletion and insertion curves of
the deletion and insertion tests of the study perform) or by marginal imputation (mean output over
the background sample, the interventional expectation of grouped Shapley
values). The Shapley value of concept k is the usual weighted average of its
marginal contributions; with at most ``max_exact_concepts`` concepts every
coalition is enumerated, otherwise antithetic permutation sampling is used.

Properties (checked in tests/test_concept_shapley.py):

* efficiency: the concept values sum to the output of the explained instance
  minus the value of the empty coalition, for every membership matrix whose
  rows sum to one;
* reduction: with the identity membership matrix and marginal imputation the
  values equal the interventional feature level Shapley values, and with a
  crisp partition they equal the grouped Shapley values of the baseline;
* symmetry and duplication invariance follow from the game being defined on
  the membership mass of a coalition, so that an exact copy of a feature that
  joins its concept changes no coalition value.
"""

from __future__ import annotations

from itertools import combinations
from math import factorial
from typing import Callable

import numpy as np

from sparc.utils import make_rng

IMPUTATIONS = ("conditional", "marginal")


def fuzzy_removal_weights(U: np.ndarray, coalition: np.ndarray) -> np.ndarray:
    """Degree to which every feature is removed when the concepts of ``coalition`` are present.

    ``coalition`` is a boolean mask of length K. Returns a vector of length d
    with entries 1 - min(1, sum of the memberships of the present concepts).
    """
    U = np.asarray(U, dtype=float)
    coalition = np.asarray(coalition, dtype=bool)
    kept = U[:, coalition].sum(axis=1) if coalition.any() else np.zeros(U.shape[0])
    return 1.0 - np.clip(kept, 0.0, 1.0)


def _coalition_outputs(model_fn, X: np.ndarray, background: np.ndarray, U: np.ndarray, coalitions: list[np.ndarray], imputation: str, k_neighbors: int) -> np.ndarray:
    """Model output of every explained instance under every coalition, shape (n_coalitions, n)."""
    from sparc.imputation import impute_removed as _impute

    n, d = X.shape
    nb = background.shape[0]
    if imputation == "conditional":
        rows = []
        for S in coalitions:
            w = fuzzy_removal_weights(U, S)
            rows.append(_impute(X, np.tile(w, (n, 1)), background, "conditional", k_neighbors=k_neighbors))
        preds = np.asarray(model_fn(np.vstack(rows)), dtype=float)
        return preds.reshape(len(coalitions), n)
    # marginal: average over the background rows of the instance with the removed part replaced by the background
    blocks = []
    for S in coalitions:
        w = fuzzy_removal_weights(U, S)
        # (n, nb, d): kept part from the instance, removed part from every background row
        Z = (1.0 - w)[None, None, :] * X[:, None, :] + w[None, None, :] * background[None, :, :]
        blocks.append(Z.reshape(n * nb, d))
    preds = np.asarray(model_fn(np.vstack(blocks)), dtype=float).reshape(len(coalitions), n, nb)
    return preds.mean(axis=2)


def concept_shapley_values(
    model_fn: Callable[[np.ndarray], np.ndarray],
    X_explain: np.ndarray,
    background: np.ndarray,
    U: np.ndarray,
    imputation: str = "conditional",
    max_exact_concepts: int = 12,
    n_permutations: int = 200,
    k_neighbors: int = 10,
    random_state: int | None = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Shapley values of fuzzy concepts for every explained instance.

    Returns the matrix of concept values with shape (n_explain, K) and the
    value of the empty coalition per instance (shape (n_explain,)), so that
    ``values.sum(axis=1) + empty`` equals the model output of the instance.
    """
    if imputation not in IMPUTATIONS:
        raise ValueError("imputation must be 'conditional' or 'marginal'")
    X = np.asarray(X_explain, dtype=float)
    background = np.asarray(background, dtype=float)
    U = np.asarray(U, dtype=float)
    n, d = X.shape
    if U.shape[0] != d:
        raise ValueError("U must have one row per feature")
    K = U.shape[1]
    if K <= max_exact_concepts:
        coalitions = []
        for r in range(K + 1):
            for S in combinations(range(K), r):
                mask = np.zeros(K, dtype=bool)
                mask[list(S)] = True
                coalitions.append(mask)
        index = {tuple(np.flatnonzero(m).tolist()): i for i, m in enumerate(coalitions)}
        v = _coalition_outputs(model_fn, X, background, U, coalitions, imputation, k_neighbors)
        weights = {s: factorial(s) * factorial(K - s - 1) / factorial(K) for s in range(K)}
        values = np.zeros((n, K))
        for key, i in index.items():
            s = len(key)
            if s == K:
                continue
            for k in range(K):
                if k in key:
                    continue
                j = index[tuple(sorted(key + (k,)))]
                values[:, k] += weights[s] * (v[j] - v[i])
        empty = v[index[()]]
        return values, empty
    # permutation sampling with antithetic pairs for large K
    rng = make_rng(random_state)
    n_pairs = max(1, int(n_permutations) // 2)
    values = np.zeros((n, K))
    empty = _coalition_outputs(model_fn, X, background, U, [np.zeros(K, dtype=bool)], imputation, k_neighbors)[0]
    for _ in range(n_pairs):
        perm = rng.permutation(K)
        for order in (perm, perm[::-1]):
            masks = []
            mask = np.zeros(K, dtype=bool)
            for k in order:
                mask = mask.copy()
                mask[k] = True
                masks.append(mask)
            v = _coalition_outputs(model_fn, X, background, U, masks, imputation, k_neighbors)
            prev = empty
            for t, k in enumerate(order):
                values[:, k] += v[t] - prev
                prev = v[t]
    return values / (2 * n_pairs), empty
