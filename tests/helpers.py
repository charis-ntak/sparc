"""Reference implementations used by the tests (copied from the study's evaluation code, which is not part of the package)."""

from __future__ import annotations

from itertools import combinations
from typing import Callable, Sequence

import numpy as np
import pandas as pd

from sparc.utils import as_index_groups, make_rng


def grouped_shapley(
    model_fn: Callable[[np.ndarray], np.ndarray],
    X_explain: np.ndarray,
    background: np.ndarray,
    groups,
    feature_names: Sequence[str] | None = None,
    max_exact_groups: int = 12,
    n_permutations: int = 200,
    random_state: int | None = 0,
) -> tuple[np.ndarray, float]:
    """Shapley values of crisp groups of features treated as players.

    The value of a coalition S is the mean model output over the background
    sample with the features of the groups in S set to the explained instance
    (interventional expectation). With at most ``max_exact_groups`` groups all
    coalitions are enumerated; otherwise Shapley values are estimated by
    permutation sampling with antithetic permutations, ``n_permutations``
    being rounded up to at least one antithetic pair.

    Returns an array of shape (n_explain, n_groups) and the expected value.
    """
    X_explain = np.asarray(X_explain, dtype=float)
    background = np.asarray(background, dtype=float)
    d = X_explain.shape[1]
    idx_groups = as_index_groups(groups, d, feature_names)
    K = len(idx_groups)
    nb = background.shape[0]
    n = X_explain.shape[0]
    expected = float(np.mean(model_fn(background)))
    out = np.zeros((n, K))
    masks = np.zeros((K, d), dtype=bool)
    for k, g in enumerate(idx_groups):
        masks[k, g] = True

    def coalition_value(x, S_mask):
        feat = masks[S_mask].any(axis=0) if S_mask.any() else np.zeros(d, dtype=bool)
        Z = background.copy()
        Z[:, feat] = x[feat]
        return float(np.mean(model_fn(Z)))

    if K <= max_exact_groups:
        from math import factorial

        coalitions = []
        for r in range(K + 1):
            coalitions.extend(combinations(range(K), r))
        weights = {}
        for S in coalitions:
            s = len(S)
            weights[s] = factorial(s) * factorial(K - s - 1) / factorial(K) if s < K else 0.0
        for i in range(n):
            x = X_explain[i]
            values = {}
            Zs = []
            for S in coalitions:
                S_mask = np.zeros(K, dtype=bool)
                S_mask[list(S)] = True
                feat = masks[S_mask].any(axis=0) if S_mask.any() else np.zeros(d, dtype=bool)
                Z = background.copy()
                Z[:, feat] = x[feat]
                Zs.append(Z)
            preds = model_fn(np.vstack(Zs)).reshape(len(coalitions), nb).mean(axis=1)
            for S, v in zip(coalitions, preds):
                values[S] = float(v)
            for k in range(K):
                phi = 0.0
                for S in coalitions:
                    if k in S:
                        continue
                    S_with = tuple(sorted(S + (k,)))
                    phi += weights[len(S)] * (values[S_with] - values[S])
                out[i, k] = phi
    else:
        rng = make_rng(random_state)
        n_pairs = max(1, int(n_permutations) // 2)
        for i in range(n):
            x = X_explain[i]
            acc = np.zeros(K)
            for _ in range(n_pairs):
                perm = rng.permutation(K)
                for order in (perm, perm[::-1]):
                    S_mask = np.zeros(K, dtype=bool)
                    prev = expected
                    for k in order:
                        S_mask[k] = True
                        cur = coalition_value(x, S_mask)
                        acc[k] += cur - prev
                        prev = cur
            out[i] = acc / (2 * n_pairs)
    return out, expected


def duplication_invariance(before: np.ndarray, after: np.ndarray) -> float:
    """Relative change of an attribution vector when a duplicate feature is added.

    ``before`` and ``after`` are aligned arrays (for example the concept
    attributions of the concept that absorbs the copy, or the feature level
    attribution of the duplicated feature). Returns mean |after - before| /
    mean |before|.
    """
    b = np.asarray(before, dtype=float)
    a = np.asarray(after, dtype=float)
    scale = float(np.mean(np.abs(b)))
    return float(np.mean(np.abs(a - b)) / scale) if scale > 1e-12 else float(np.mean(np.abs(a - b)))


def sign_confidence_calibration(stated: np.ndarray, agreement: np.ndarray, n_bins: int = 10) -> dict:
    """Reliability diagram of stated sign confidence against observed sign agreement.

    Returns the binned table and the expected calibration error (ECE), the
    weighted mean absolute gap between stated confidence and observed frequency.
    """
    s = np.asarray(stated, dtype=float).ravel()
    a = np.asarray(agreement, dtype=float).ravel()
    edges = np.linspace(0.5, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(s, edges) - 1, 0, n_bins - 1)
    rows = []
    ece = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.sum() == 0:
            rows.append({"bin_low": edges[b], "bin_high": edges[b + 1], "count": 0, "mean_confidence": np.nan, "observed": np.nan})
            continue
        mc, ob = float(s[m].mean()), float(a[m].mean())
        rows.append({"bin_low": edges[b], "bin_high": edges[b + 1], "count": int(m.sum()), "mean_confidence": mc, "observed": ob})
        ece += m.sum() / s.size * abs(mc - ob)
    return {"table": pd.DataFrame(rows), "ece": float(ece)}
