"""Imputation of removed features (conditional nearest neighbour mean or marginal replacement)."""

from __future__ import annotations

import numpy as np


def impute_removed(
    X_query: np.ndarray,
    weights: np.ndarray,
    background: np.ndarray,
    mode: str,
    rng: np.random.Generator | None = None,
    k_neighbors: int = 10,
    replacement: np.ndarray | None = None,
) -> np.ndarray:
    """Replace features according to removal weights in [0, 1].

    ``weights`` has shape (n, d): the degree to which every feature of every
    row is removed. With 'conditional' imputation the replacement values are
    the mean of the ``k_neighbors`` nearest background rows, measured on the
    kept features (weighted by 1 - w) after division by the standard deviation
    of every background column. Columns that are constant in the background
    carry no information about the neighbors and are left out of the
    distance, so that a query value differing from the constant cannot swamp
    the kept features; when no kept feature is informative the background
    mean is used. With 'marginal' imputation the replacement rows are
    ``replacement`` (shape (n, d)) when given, so that a caller can reuse the
    same rows across calls, and otherwise one background row drawn at random
    with ``rng`` for every query row.
    """
    n, d = X_query.shape
    if mode == "marginal":
        if replacement is None:
            if rng is None:
                raise ValueError("marginal imputation requires rng or replacement rows")
            replacement = background[rng.integers(0, background.shape[0], size=n)]
        rep = np.asarray(replacement, dtype=float)
    elif mode == "conditional":
        rep = np.empty_like(X_query)
        sd = background.std(axis=0)
        informative = sd > 1e-12
        sd = np.where(informative, sd, 1.0)
        Bz = background / sd
        k = min(k_neighbors, background.shape[0])
        for i in range(n):
            keep_w = (1.0 - weights[i]) * informative
            if keep_w.sum() <= 1e-9:
                rep[i] = background.mean(axis=0)
                continue
            diff = (Bz - X_query[i] / sd) * np.sqrt(keep_w)
            dist = (diff**2).sum(axis=1)
            nn = np.argpartition(dist, k - 1)[:k]
            rep[i] = background[nn].mean(axis=0)
    else:
        raise ValueError("mode must be 'conditional' or 'marginal'")
    return (1.0 - weights) * X_query + weights * rep


_impute = impute_removed
