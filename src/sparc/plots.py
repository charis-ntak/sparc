"""Plots of SPARC explanations.

Every function takes an :class:`sparc.Explanation` (or a subset of one,
``explanation[i]``), draws with matplotlib on the given or a new axes and
returns the axes; ``show=True`` calls ``plt.show()``. The plots follow the
conventions that users of attribution libraries know, at the concept level:

``bar``        mean absolute attribution of every concept, or the signed
               attribution of one instance, with the support of the fuzzy
               number as a whisker;
``waterfall``  from the base value to the model output of one instance,
               concept by concept;
``beeswarm``   one row per concept, one dot per instance, coloured by the
               concept score of the instance;
``heatmap``    instances by concepts, instances ordered by the model output;
``dependence`` attribution of a concept against its concept score, coloured
               by the concept with the strongest interaction;
``force``      the contributions of one instance pushing the output away from
               the base value;
``decision``   the cumulative path of every instance from the base value to
               its output through the concepts.

The views that only SPARC can draw:

``interval_bars``        point, core and support of every attribution with the
                         sign confidence and the model class disagreement;
``linguistic_profile``   the degree of every linguistic label per concept;
``disagreement``         the attribution distribution of every model class;
``concept_graph``        the fuzzy partition as a picture of the features;
``front``                the Pareto front with the knee, the fidelity optimal
                         member and the current explanation;
``front_navigator``      the explanation along the front, member by member;
``selection_membership`` how often the front retains every concept and feature;
``coverage``             coverage of a replicated attribution by the intervals.

``fuzzy_attribution``, ``linguistic_bars``, ``pareto_front`` and ``membership``
of :mod:`sparc.plotting` are re-exported here.
"""

from __future__ import annotations

import numpy as np

from sparc.plotting import GRID, MUTED, PALETTE, TEXT, _plt, style_axes
from sparc.plotting import plot_fuzzy_attribution as fuzzy_attribution
from sparc.plotting import plot_linguistic_bars as linguistic_bars
from sparc.plotting import plot_membership as membership
from sparc.plotting import plot_pareto_front as pareto_front

POSITIVE = PALETTE[0]
NEGATIVE = PALETTE[1]

__all__ = [
    "bar", "waterfall", "beeswarm", "heatmap", "dependence", "force", "decision",
    "interval_bars", "linguistic_profile", "disagreement", "concept_graph", "front", "front_navigator", "selection_membership", "coverage",
    "fuzzy_attribution", "linguistic_bars", "pareto_front", "membership",
]


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------


def _axes(ax, figsize):
    plt = _plt()
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    return plt, ax


def _finish(plt, ax, show):
    style_axes(ax)
    if show:
        plt.show()
    return ax


def _concept_index(explanation, concept) -> int:
    if isinstance(concept, (int, np.integer)):
        k = int(concept)
        if not 0 <= k < explanation.n_concepts:
            raise IndexError(f"concept {concept} out of range for {explanation.n_concepts} concepts")
        return k
    names = explanation.concept_names
    if concept in names:
        return names.index(concept)
    raise KeyError(f"unknown concept {concept!r}; the concepts are {names}")


def _ordered_concepts(explanation, max_display: int | None, values: np.ndarray | None = None) -> np.ndarray:
    """Concept indices ordered by importance (mean absolute attribution), most important first, at most max_display."""
    imp = np.abs(values).mean(axis=0) if values is not None else explanation.importance
    order = np.argsort(-imp, kind="stable")
    if max_display is not None and max_display < len(order):
        order = order[:max_display]
    return order


def _tick_labels(explanation, order) -> list[str]:
    retained = set(explanation.retained.tolist())
    return [explanation.concept_names[k] + (" *" if k in retained else "") for k in order]


def _instance_label(explanation, instance: int) -> int:
    return int(explanation.instance_index[instance]) if explanation.instance_index is not None else int(instance)


def _single(explanation, instance):
    """Instance index to draw for a per instance plot: the given one, or the only one of a one instance explanation."""
    if instance is None:
        if explanation.n_explain != 1:
            raise ValueError("this plot draws one instance: pass instance=i or index the explanation first (explanation[i])")
        return 0
    return int(instance)


# ----------------------------------------------------------------------
# parity plots
# ----------------------------------------------------------------------


def bar(explanation, max_display: int = 10, instance: int | None = None, ax=None, show: bool = False):
    """Concept importance bars.

    Without ``instance`` (and with more than one explained instance) the bars
    are the mean absolute attribution of every concept over the explained
    instances, with the support of the fuzzy global importance as a whisker.
    For one instance the bars are the signed attributions with the support of
    the fuzzy attribution. Retained concepts are marked with an asterisk.
    """
    plt, ax = _axes(ax, (6.5, 0.45 * min(max_display, explanation.n_concepts) + 1.2))
    if instance is None and explanation.n_explain == 1:
        instance = 0
    if instance is None:
        order = _ordered_concepts(explanation, max_display)
        values = explanation.importance[order]
        lo = explanation.global_quantiles[order, 0]
        hi = explanation.global_quantiles[order, 4]
        colors = [POSITIVE] * len(order)
        xlabel = "mean |attribution| over instances (whisker: support of the global importance)"
    else:
        v = explanation.values[instance]
        order = np.argsort(-np.abs(v), kind="stable")[:max_display]
        values = v[order]
        lo = explanation.quantiles[instance, order, 0]
        hi = explanation.quantiles[instance, order, 4]
        colors = [POSITIVE if x >= 0 else NEGATIVE for x in values]
        xlabel = f"attribution of instance {_instance_label(explanation, instance)} (whisker: support of the fuzzy attribution)"
    y = np.arange(len(order))[::-1]
    ax.barh(y, values, color=colors, height=0.6)
    ax.hlines(y, lo, hi, color=TEXT, linewidth=1)
    ax.vlines(np.concatenate([lo, hi]), np.concatenate([y, y]) - 0.12, np.concatenate([y, y]) + 0.12, color=TEXT, linewidth=1)
    for yi, val in zip(y, values):
        ax.text(val, yi, f" {val:+.3g}" if val >= 0 else f"{val:+.3g} ", va="center", ha="left" if val >= 0 else "right", fontsize=8, color=MUTED)
    ax.set_yticks(y)
    ax.set_yticklabels(_tick_labels(explanation, order))
    ax.axvline(0, color=MUTED, linewidth=0.8)
    ax.set_xlabel(xlabel)
    return _finish(plt, ax, show)


def waterfall(explanation, instance: int | None = None, max_display: int = 10, ax=None, show: bool = False):
    """From the base value to the model output of one instance, one step per concept, the smallest steps collapsed."""
    i = _single(explanation, instance)
    v = explanation.values[i]
    base = float(explanation.base_values[i])
    out = float(explanation.output[i])
    order = np.argsort(-np.abs(v), kind="stable")
    shown = list(order[:max_display])
    rest = list(order[max_display:])
    steps = [(explanation.concept_names[k] + (" *" if k in set(explanation.retained.tolist()) else ""), float(v[k])) for k in shown]
    if rest:
        steps.append((f"{len(rest)} other concepts", float(v[rest].sum())))
    plt, ax = _axes(ax, (6.5, 0.45 * len(steps) + 1.4))
    # draw from the top (largest) down, cumulative from the base value
    cum = base
    y = np.arange(len(steps))[::-1]
    for yi, (name, val) in zip(y, steps):
        color = POSITIVE if val >= 0 else NEGATIVE
        ax.barh(yi, val, left=cum, color=color, height=0.6)
        ax.text(cum + val, yi, f" {val:+.3g}" if val >= 0 else f"{val:+.3g} ", va="center", ha="left" if val >= 0 else "right", fontsize=8, color=MUTED)
        cum += val
    ax.axvline(base, color=MUTED, linewidth=0.8, linestyle="--")
    ax.axvline(out, color=TEXT, linewidth=0.8)
    ax.text(base, len(steps) - 0.45, f"base {base:.3g}", ha="center", va="bottom", fontsize=8, color=MUTED)
    ax.text(out, len(steps) + 0.05, f"output {out:.3g}", ha="center", va="bottom", fontsize=8, color=TEXT)
    ax.set_ylim(-0.6, len(steps) + 0.6)
    ax.set_yticks(y)
    ax.set_yticklabels([s[0] for s in steps])
    ax.set_xlabel(f"model output, instance {_instance_label(explanation, i)}")
    return _finish(plt, ax, show)


def _swarm_offsets(x: np.ndarray, n_bins: int = 40, max_offset: float = 0.4) -> np.ndarray:
    """Vertical offsets that stack points of close x values, as a simple swarm."""
    n = x.size
    if n == 0:
        return np.zeros(0)
    lo, hi = float(np.min(x)), float(np.max(x))
    if hi - lo < 1e-12:
        bins = np.zeros(n, dtype=int)
    else:
        bins = np.minimum(((x - lo) / (hi - lo) * n_bins).astype(int), n_bins - 1)
    offsets = np.zeros(n)
    for b in np.unique(bins):
        idx = np.flatnonzero(bins == b)
        m = idx.size
        ranks = np.arange(m)
        # alternate above and below the row centre, denser bins spread wider
        signs = np.where(ranks % 2 == 0, 1.0, -1.0)
        levels = (ranks + 1) // 2
        spread = min(max_offset, 0.08 * m)
        offsets[idx[np.argsort(x[idx])]] = signs * levels / max(1, levels.max()) * spread
    return offsets


def beeswarm(explanation, max_display: int = 10, ax=None, show: bool = False, cmap: str = "coolwarm"):
    """One row per concept, one dot per instance at its attribution, coloured by the instance's concept score."""
    plt, ax = _axes(ax, (6.8, 0.5 * min(max_display, explanation.n_concepts) + 1.2))
    order = _ordered_concepts(explanation, max_display)
    V = explanation.values
    D = explanation.data
    y = np.arange(len(order))[::-1]
    sc = None
    for yi, k in zip(y, order):
        x = V[:, k]
        off = _swarm_offsets(x)
        if D is not None:
            c = D[:, k]
            lo, hi = np.percentile(c, [5, 95])
            c = np.clip((c - lo) / (hi - lo if hi > lo else 1.0), 0.0, 1.0)
            sc = ax.scatter(x, yi + off, c=c, cmap=cmap, vmin=0.0, vmax=1.0, s=14, alpha=0.85, linewidths=0)
        else:
            ax.scatter(x, yi + off, color=POSITIVE, s=14, alpha=0.85, linewidths=0)
    ax.set_yticks(y)
    ax.set_yticklabels(_tick_labels(explanation, order))
    ax.axvline(0, color=MUTED, linewidth=0.8)
    ax.set_xlabel("attribution to model output")
    if sc is not None:
        cb = plt.colorbar(sc, ax=ax, ticks=[0, 1], pad=0.02, aspect=30)
        cb.ax.set_yticklabels(["low", "high"])
        cb.set_label("concept score", rotation=270, labelpad=10)
        cb.outline.set_visible(False)
    return _finish(plt, ax, show)


def heatmap(explanation, max_display: int = 10, instance_order: str = "output", ax=None, show: bool = False, cmap: str = "coolwarm"):
    """Attributions of every instance (columns) and concept (rows), with the model output above and the importance beside."""
    plt, ax = _axes(ax, (7.5, 0.4 * min(max_display, explanation.n_concepts) + 2.2))
    order = _ordered_concepts(explanation, max_display)
    V = explanation.values[:, order]
    out = explanation.output
    if instance_order == "output":
        cols = np.argsort(out, kind="stable")
    elif instance_order == "none":
        cols = np.arange(explanation.n_explain)
    else:
        raise ValueError("instance_order must be 'output' or 'none'")
    M = V[cols].T
    vmax = float(np.abs(M).max()) if M.size else 1.0
    im = ax.imshow(M, aspect="auto", cmap=cmap, vmin=-vmax, vmax=vmax, interpolation="nearest")
    ax.set_yticks(np.arange(len(order)))
    ax.set_yticklabels(_tick_labels(explanation, order))
    ax.set_xlabel("instances, ordered by model output" if instance_order == "output" else "instances")
    ax.set_xticks([])
    # the model output as a line above the matrix
    top = ax.inset_axes([0.0, 1.02, 1.0, 0.18], sharex=ax)
    top.plot(np.arange(len(cols)), out[cols], color=TEXT, linewidth=1)
    top.set_ylabel("output", fontsize=8)
    top.tick_params(labelbottom=False, labelsize=7)
    top.grid(False)
    for side in ("top", "right"):
        top.spines[side].set_visible(False)
    # the importance beside the matrix
    right = ax.inset_axes([1.02, 0.0, 0.12, 1.0], sharey=ax)
    right.barh(np.arange(len(order)), np.abs(V).mean(axis=0), color=POSITIVE, height=0.7)
    right.tick_params(labelleft=False, labelsize=7)
    right.set_xlabel("mean |attr.|", fontsize=8)
    right.grid(False)
    for side in ("top", "right"):
        right.spines[side].set_visible(False)
    cb = plt.colorbar(im, ax=ax, pad=0.18, aspect=30)
    cb.set_label("attribution", rotation=270, labelpad=10)
    cb.outline.set_visible(False)
    ax.grid(False)
    if show:
        plt.show()
    return ax


def _interaction_partner(explanation, k: int) -> int | None:
    """Concept whose score explains most of the variation of concept k's attribution around its trend."""
    D, V = explanation.data, explanation.values
    if D is None or explanation.n_concepts < 2 or explanation.n_explain < 4:
        return None
    x, y = D[:, k], V[:, k]
    if np.std(x) < 1e-12:
        return None
    coef = np.polyfit(x, y, 1)
    resid = y - np.polyval(coef, x)
    best, best_r = None, 0.0
    for j in range(explanation.n_concepts):
        if j == k or np.std(D[:, j]) < 1e-12 or np.std(resid) < 1e-12:
            continue
        r = abs(float(np.corrcoef(D[:, j], resid)[0, 1]))
        if r > best_r:
            best, best_r = j, r
    return best


def dependence(explanation, concept, color=None, ax=None, show: bool = False, cmap: str = "coolwarm"):
    """Attribution of one concept against its concept score, coloured by the score of an interacting concept.

    ``concept`` and ``color`` are concept names or indices; ``color=None``
    picks the concept whose score correlates most with the residual of the
    linear trend, ``color=False`` draws without colour.
    """
    k = _concept_index(explanation, concept)
    D = explanation.data
    if D is None:
        raise ValueError("the dependence plot needs the explained instances (explanation.X is None)")
    plt, ax = _axes(ax, (6.0, 4.2))
    x, y = D[:, k], explanation.values[:, k]
    j = None
    if color is None:
        j = _interaction_partner(explanation, k)
    elif color is not False:
        j = _concept_index(explanation, color)
    if j is not None:
        c = D[:, j]
        sc = ax.scatter(x, y, c=c, cmap=cmap, s=18, alpha=0.9, linewidths=0)
        cb = plt.colorbar(sc, ax=ax, pad=0.02, aspect=30)
        cb.set_label(f"score of {explanation.concept_names[j]}", rotation=270, labelpad=12)
        cb.outline.set_visible(False)
    else:
        ax.scatter(x, y, color=POSITIVE, s=18, alpha=0.9, linewidths=0)
    ax.axhline(0, color=MUTED, linewidth=0.8)
    ax.set_xlabel(f"concept score of {explanation.concept_names[k]} (standardized, membership weighted)")
    ax.set_ylabel(f"attribution of {explanation.concept_names[k]}")
    return _finish(plt, ax, show)


def force(explanation, instance: int | None = None, max_display: int = 8, ax=None, show: bool = False):
    """The contributions of one instance as segments pushing the output away from the base value."""
    i = _single(explanation, instance)
    v = explanation.values[i]
    base = float(explanation.base_values[i])
    out = float(explanation.output[i])
    names = explanation.concept_names
    order = np.argsort(-np.abs(v), kind="stable")
    shown = list(order[:max_display])
    rest_sum = float(v[order[max_display:]].sum()) if len(order) > max_display else 0.0
    pos = [(names[k], float(v[k])) for k in shown if v[k] >= 0]
    neg = [(names[k], float(v[k])) for k in shown if v[k] < 0]
    if rest_sum >= 0 and rest_sum != 0.0:
        pos.append(("other", rest_sum))
    elif rest_sum < 0:
        neg.append(("other", rest_sum))
    pos.sort(key=lambda t: -t[1])
    neg.sort(key=lambda t: t[1])
    plt, ax = _axes(ax, (8.0, 2.2))
    # positives are laid from the output leftwards towards the base, negatives from the output rightwards
    total_pos = sum(t[1] for t in pos)
    left = out - total_pos
    x = left
    for name, val in pos:
        ax.barh(0, val, left=x, color=POSITIVE, height=0.5, edgecolor="white", linewidth=0.8)
        if val / max(abs(out - base), 1e-12) > 0.06:
            ax.text(x + val / 2, 0, f"{name}\n{val:+.3g}", ha="center", va="center", fontsize=7, color="white")
        x += val
    x = out
    for name, val in neg:
        ax.barh(0, -val, left=x, color=NEGATIVE, height=0.5, edgecolor="white", linewidth=0.8)
        if -val / max(abs(out - base), 1e-12) > 0.06:
            ax.text(x - val / 2, 0, f"{name}\n{val:+.3g}", ha="center", va="center", fontsize=7, color="white")
        x -= val
    ax.axvline(base, color=MUTED, linewidth=0.8, linestyle="--")
    ax.axvline(out, color=TEXT, linewidth=1.2)
    ax.text(base, 0.42, f"base {base:.3g}", ha="center", va="bottom", fontsize=8, color=MUTED)
    ax.text(out, 0.42, f"output {out:.3g}", ha="center", va="bottom", fontsize=8, color=TEXT)
    ax.set_yticks([])
    ax.set_ylim(-0.5, 0.75)
    ax.set_xlabel(f"model output, instance {_instance_label(explanation, i)} (blue pushes up, orange pushes down)")
    ax.grid(False)
    for side in ("left",):
        ax.spines[side].set_visible(False)
    return _finish(plt, ax, show)


def decision(explanation, max_display: int = 10, instances=None, ax=None, show: bool = False, cmap: str = "coolwarm"):
    """Cumulative path of every instance from the base value to its output through the concepts, ordered by importance."""
    plt, ax = _axes(ax, (6.0, 0.45 * min(max_display, explanation.n_concepts) + 1.6))
    order = _ordered_concepts(explanation, max_display)
    rest = [k for k in range(explanation.n_concepts) if k not in set(order.tolist())]
    idx = np.arange(explanation.n_explain) if instances is None else np.asarray(instances, dtype=int)
    V = explanation.values[idx]
    base = explanation.base_values[idx]
    out = explanation.output[idx]
    # rows from the least important (bottom) to the most important (top); the rest is folded into the first step
    rows = list(order[::-1])
    labels = _tick_labels(explanation, rows)
    start = base + (V[:, rest].sum(axis=1) if rest else 0.0)
    cum = np.cumsum(V[:, rows], axis=1) + start[:, None]
    paths = np.concatenate([start[:, None], cum], axis=1)
    y = np.arange(len(rows) + 1)
    lo, hi = float(out.min()), float(out.max())
    norm = (out - lo) / (hi - lo) if hi > lo else np.full(out.shape, 0.5)
    colors = plt.get_cmap(cmap)(norm)
    for p, c in zip(paths, colors):
        ax.plot(p, y, color=c, linewidth=0.9, alpha=0.8)
    ax.axvline(float(np.mean(base)), color=MUTED, linewidth=0.8, linestyle="--")
    ax.set_yticks(y)
    ax.set_yticklabels(["base" + (f" + {len(rest)} other" if rest else "")] + labels)
    ax.set_xlabel("model output")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(lo, hi if hi > lo else lo + 1.0))
    cb = plt.colorbar(sm, ax=ax, pad=0.02, aspect=30)
    cb.set_label("model output", rotation=270, labelpad=10)
    cb.outline.set_visible(False)
    return _finish(plt, ax, show)


# ----------------------------------------------------------------------
# views that only SPARC can draw
# ----------------------------------------------------------------------


def interval_bars(explanation, instance: int | None = None, max_display: int = 10, ax=None, show: bool = False, compact: bool = False):
    """Attribution of one instance with its core and support, the sign confidence and the model class disagreement.

    The bar is the point attribution, the thick line the core of the fuzzy
    number (central 50 percent of the pool by default), the thin line its
    support (central 90 percent). Where the model classes disagree beyond the
    threshold the attribution is of type 2 and the centroid of every model
    class is marked with a tick. The sign confidence and the disagreement
    index are written beside every bar.
    """
    i = _single(explanation, instance)
    v = explanation.values[i]
    order = np.argsort(-np.abs(v), kind="stable")[:max_display]
    plt, ax = _axes(ax, (7.5, 0.5 * len(order) + 1.3))
    y = np.arange(len(order))[::-1]
    B, M = explanation.attribution.n_resamples, explanation.attribution.n_model_classes
    pool = explanation.concept_pool.reshape(B, M, explanation.n_explain, explanation.n_concepts)
    for yi, k in zip(y, order):
        q = explanation.quantiles[i, k]
        color = POSITIVE if v[k] >= 0 else NEGATIVE
        ax.barh(yi, v[k], color=color, height=0.5, alpha=0.9)
        ax.hlines(yi, q[0], q[4], color=TEXT, linewidth=1.0)
        ax.hlines(yi, q[1], q[3], color=TEXT, linewidth=4.0, alpha=0.8)
        if explanation.type2_mask[i, k]:
            for m in range(M):
                ax.plot([float(np.median(pool[:, m, i, k]))], [yi], marker="|", color=PALETTE[3], markersize=14, markeredgewidth=2)
        flag = " (type 2)" if explanation.type2_mask[i, k] else ""
        if compact:
            text = f"sign conf. {explanation.sign_confidence[i, k]:.2f}{flag}"
        else:
            text = f"sign conf. {explanation.sign_confidence[i, k]:.2f}, disagreement {explanation.disagreement[i, k]:.2f}{flag}"
        ax.text(max(q[4], v[k], 0.0), yi, "  " + text, va="center", ha="left", fontsize=7.5, color=MUTED)
    ax.set_yticks(y)
    ax.set_yticklabels(_tick_labels(explanation, order))
    ax.axvline(0, color=MUTED, linewidth=0.8)
    detail = "" if compact else "\n(bar: point, thick: core, thin: support, ticks: model class medians)"
    ax.set_xlabel(f"attribution of instance {_instance_label(explanation, i)}{detail}")
    xlim = ax.get_xlim()
    ax.set_xlim(xlim[0], xlim[1] + 0.7 * (xlim[1] - xlim[0]))
    return _finish(plt, ax, show)


def linguistic_profile(explanation, instance: int | None = None, max_display: int = 10, ax=None, show: bool = False, cmap: str = "Blues"):
    """Degree of every linguistic label for every concept of one instance, the best label marked."""
    i = _single(explanation, instance)
    v = explanation.values[i]
    order = np.argsort(-np.abs(v), kind="stable")[:max_display]
    names = explanation.label_set.names
    plt, ax = _axes(ax, (0.9 * len(names) + 2.5, 0.45 * len(order) + 1.2))
    P = explanation.linguistic[i][order]
    im = ax.imshow(P, aspect="auto", cmap=cmap, vmin=0.0, vmax=1.0, interpolation="nearest")
    best = P.argmax(axis=1)
    for r, c in enumerate(best):
        ax.text(c, r, f"{P[r, c]:.2f}", ha="center", va="center", fontsize=8, color="white" if P[r, c] > 0.55 else TEXT, fontweight="bold")
    ax.set_xticks(np.arange(len(names)))
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_yticks(np.arange(len(order)))
    ax.set_yticklabels(_tick_labels(explanation, order))
    ax.set_xlabel(f"linguistic labels, instance {_instance_label(explanation, i)} (cell: compatibility of the fuzzy attribution with the label)")
    cb = plt.colorbar(im, ax=ax, pad=0.02, aspect=30)
    cb.set_label("compatibility", rotation=270, labelpad=10)
    cb.outline.set_visible(False)
    ax.grid(False)
    if show:
        plt.show()
    return ax


def disagreement(explanation, instance: int | None = None, max_display: int = 8, ax=None, show: bool = False):
    """Attribution distribution of every model class, concept by concept, with the disagreement index.

    For one instance (``instance`` or a one instance explanation) the dots are
    the resamples of every model class; without an instance the dots are the
    mean absolute attributions over the explained instances per resample and
    model class. The model dependence of an attribution is what no feature
    level method reports.
    """
    B, M = explanation.attribution.n_resamples, explanation.attribution.n_model_classes
    pool = explanation.concept_pool.reshape(B, M, explanation.n_explain, explanation.n_concepts)
    if instance is None and explanation.n_explain == 1:
        instance = 0
    if instance is None:
        values = np.abs(pool).mean(axis=2)  # (B, M, K)
        order = _ordered_concepts(explanation, max_display)
        xlabel = "mean |attribution| over instances, per resample and model class"
        dis = None
    else:
        values = pool[:, :, int(instance)]  # (B, M, K)
        order = np.argsort(-np.abs(explanation.values[int(instance)]), kind="stable")[:max_display]
        xlabel = f"attribution of instance {_instance_label(explanation, int(instance))}, per resample and model class"
        dis = explanation.disagreement[int(instance)]
    classes = list(explanation.attribution.model_class_names)
    plt, ax = _axes(ax, (7.0, 0.32 * M * len(order) + 1.4))
    yticks, ylabels = [], []
    row = 0
    gap = 0.6
    for k in order:
        centre = []
        for m in range(M):
            yv = row
            x = values[:, m, k]
            jitter = (np.random.default_rng(m).random(x.size) - 0.5) * 0.3
            ax.scatter(x, yv + jitter, s=12, color=PALETTE[m % len(PALETTE)], alpha=0.75, linewidths=0, label=classes[m] if k == order[0] else None)
            ax.plot([np.median(x)], [yv], marker="|", color=TEXT, markersize=12, markeredgewidth=1.5)
            centre.append(yv)
            row += 1
        ymid = float(np.mean(centre))
        yticks.append(ymid)
        ylabels.append(explanation.concept_names[k] + (" *" if k in set(explanation.retained.tolist()) else "") + (f"\ndisagreement {dis[k]:.2f}" if dis is not None else ""))
        if k != order[-1]:
            ax.axhline(row - 1 + gap / 2 + 0.2, color=GRID, linewidth=0.8)
        row += gap
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels)
    ax.invert_yaxis()
    ax.axvline(0, color=MUTED, linewidth=0.8)
    ax.set_xlabel(xlabel)
    ax.legend(fontsize=8, title="model class", title_fontsize=8, loc="best", frameon=False)
    return _finish(plt, ax, show)


def concept_graph(explanation, min_correlation: float = 0.3, ax=None, show: bool = False, seed: int = 0):
    """Features placed by the similarity of their correlation profiles, coloured by concept with membership as opacity.

    Edges join features whose absolute correlation (over the explained
    instances, or over the training sample statistics when available) is at
    least ``min_correlation``; the node size follows the mean absolute feature
    level attribution. A fuzzy partition shows as mixed colours and as nodes
    between the groups.
    """
    plt, ax = _axes(ax, (6.5, 5.5))
    X = explanation.X
    U = explanation.partition.U
    d, K = U.shape
    if X is not None and X.shape[0] >= 4:
        with np.errstate(invalid="ignore", divide="ignore"):
            C = np.corrcoef(X, rowvar=False)
        C = np.nan_to_num(C)
    else:
        C = U @ U.T
    # two dimensional layout: classical multidimensional scaling of the correlation profile distances
    D2 = ((C[:, None, :] - C[None, :, :]) ** 2).sum(axis=2)
    J = np.eye(d) - np.ones((d, d)) / d
    Bm = -0.5 * J @ D2 @ J
    w, V = np.linalg.eigh(Bm)
    idx = np.argsort(w)[::-1][:2]
    coords = V[:, idx] * np.sqrt(np.clip(w[idx], 0.0, None))
    if np.allclose(coords, 0):
        ang = np.linspace(0, 2 * np.pi, d, endpoint=False)
        coords = np.column_stack([np.cos(ang), np.sin(ang)])
    coords = coords + np.random.default_rng(seed).normal(scale=1e-3, size=coords.shape)
    coords = _spread(coords, min_distance=0.12)
    # edges
    for a in range(d):
        for b in range(a + 1, d):
            r = abs(float(C[a, b]))
            if r >= min_correlation:
                ax.plot(coords[[a, b], 0], coords[[a, b], 1], color=MUTED, alpha=0.25 + 0.6 * (r - min_correlation) / max(1e-9, 1 - min_correlation), linewidth=0.5 + 2.5 * r, zorder=1)
    # nodes: colour mixed by membership, size by feature level importance
    palette = np.array([_hex_to_rgb(PALETTE[k % len(PALETTE)]) for k in range(K)])
    colors = U @ palette
    imp = np.abs(explanation.feature_attribution()).mean(axis=0)
    size = 60 + 340 * imp / (imp.max() if imp.max() > 0 else 1.0)
    dominant = U.argmax(axis=1)
    alpha = 0.45 + 0.55 * U.max(axis=1)
    for j in range(d):
        ax.scatter(coords[j, 0], coords[j, 1], s=size[j], color=np.clip(colors[j], 0, 1), alpha=float(alpha[j]), edgecolor=TEXT, linewidth=0.6, zorder=3)
        ax.text(coords[j, 0], coords[j, 1], explanation.feature_names[j], ha="center", va="center", fontsize=7, color=TEXT, zorder=4)
    for k in range(K):
        members = np.flatnonzero(dominant == k)
        if members.size:
            cx, cy = (coords[members] * U[members, k][:, None]).sum(axis=0) / U[members, k].sum()
            ax.text(cx, cy, explanation.concept_names[k], fontsize=9, color=PALETTE[k % len(PALETTE)], ha="center", va="bottom", fontweight="bold", zorder=5,
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.7))
    handles = [plt.Line2D([], [], marker="o", linestyle="", color=PALETTE[k % len(PALETTE)], markersize=8, label=explanation.concept_names[k]) for k in range(K)]
    ax.legend(handles=handles, fontsize=8, frameon=False, loc="best")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("features placed by the similarity of their correlation profiles\ncolour: concept membership, size: mean |attribution|, edges: |correlation|")
    ax.margins(0.12)
    ax.grid(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_visible(False)
    if show:
        plt.show()
    return ax


def _spread(coords: np.ndarray, min_distance: float = 0.1, iterations: int = 200) -> np.ndarray:
    """Scale the layout to the unit square and push apart nodes closer than ``min_distance`` (a light repulsion)."""
    P = np.asarray(coords, dtype=float).copy()
    span = P.max(axis=0) - P.min(axis=0)
    span = np.where(span > 1e-12, span, 1.0)
    P = (P - P.min(axis=0)) / span
    for _ in range(iterations):
        moved = False
        for a in range(len(P)):
            for b in range(a + 1, len(P)):
                diff = P[a] - P[b]
                dist = float(np.hypot(*diff))
                if dist < min_distance:
                    if dist < 1e-9:
                        diff = np.array([np.cos(a + b), np.sin(a - b)])
                        dist = 1.0
                    step = 0.5 * (min_distance - dist) * diff / dist
                    P[a] += step
                    P[b] -= step
                    moved = True
        if not moved:
            break
    return P


def _hex_to_rgb(h: str) -> np.ndarray:
    h = h.lstrip("#")
    return np.array([int(h[i : i + 2], 16) / 255.0 for i in (0, 2, 4)])


def front(explanation, ax=None, show: bool = False, annotate: bool = True):
    """The Pareto front of explanation configurations with the knee, the fidelity optimal member and the current one.

    Fidelity loss against complexity, coloured by instability; dominated
    configurations in grey. The knee is the default explanation, the
    fidelity optimal member the most faithful one, and the current
    configuration is marked when the explanation was built at another front
    member with ``explanation.at``.
    """
    plt, ax = _axes(ax, (6.0, 4.2))
    table = explanation.configurations
    dom = table[~table["pareto"]]
    fr = table[table["pareto"]].sort_values("complexity")
    ax.scatter(dom["complexity"], dom["fidelity_loss"], s=14, color=GRID, edgecolor=MUTED, linewidth=0.4, label="dominated", zorder=1)
    vmax = max(0.05, float(fr["instability"].max()))
    sc = ax.scatter(fr["complexity"], fr["fidelity_loss"], s=40, c=fr["instability"], cmap="Blues", vmin=0, vmax=vmax, edgecolor=TEXT, linewidth=0.5, label="Pareto front", zorder=2)
    ax.plot(fr["complexity"], fr["fidelity_loss"], color=POSITIVE, linewidth=1, alpha=0.6, zorder=1)
    knee = table.loc[explanation.selection.knee]
    ax.scatter([knee["complexity"]], [knee["fidelity_loss"]], s=160, facecolor="none", edgecolor=NEGATIVE, linewidth=2, label="knee (default)", zorder=3)
    fo = table.loc[explanation.fidelity_optimal]
    ax.scatter([fo["complexity"]], [fo["fidelity_loss"]], s=160, marker="s", facecolor="none", edgecolor=PALETTE[2], linewidth=2, label="fidelity optimal", zorder=3)
    if explanation.configuration_index is not None and explanation.configuration_index not in (explanation.selection.knee,):
        cur = table.loc[explanation.configuration_index]
        ax.scatter([cur["complexity"]], [cur["fidelity_loss"]], s=220, marker="D", facecolor="none", edgecolor=PALETTE[3], linewidth=2, label="this explanation", zorder=3)
    if annotate:
        for _, row in explanation.selection.distinct_front.iterrows():
            ax.annotate(f"K={int(row['n_concepts'])}, r={int(row['n_retained'])}" if "n_retained" in row else f"K={int(row['n_concepts'])}", (row["complexity"], row["fidelity_loss"]), textcoords="offset points", xytext=(5, 4), fontsize=7, color=MUTED)
    cb = plt.colorbar(sc, ax=ax, pad=0.02, aspect=30)
    cb.set_label("instability", rotation=270, labelpad=10)
    cb.outline.set_visible(False)
    ax.set_xlabel("complexity (retained concepts weighted by fuzziness)")
    ax.set_ylabel("fidelity loss (1 - cross validated R2 of the surrogate)")
    ax.legend(fontsize=8, frameon=False)
    return _finish(plt, ax, show)


def front_navigator(explanation, instance: int | None = None, n_members: int = 4, max_display: int = 6, axes=None, show: bool = False):
    """Small multiples of the explanation along the front, from the simplest to the most faithful member.

    Each panel is the interval bar view of the chosen instance (or the global
    importance when ``instance`` is None and the explanation holds several
    instances) at one distinct front member, with the member's objectives in
    the title. The knee and the fidelity optimal member are always included.
    """
    members = explanation.front_members(distinct=True)
    keep = {explanation.selection.knee, explanation.fidelity_optimal}
    if len(members) > n_members:
        picks = [members[0], members[-1]] + [members[int(round(t))] for t in np.linspace(0, len(members) - 1, n_members)[1:-1]]
        chosen = sorted(set(picks) | keep, key=members.index)
        if len(chosen) > n_members:
            # keep the extremes and the mandatory ones, drop the middle picks
            extra = [m for m in chosen if m not in keep and m not in (members[0], members[-1])]
            while len(chosen) > n_members and extra:
                chosen.remove(extra.pop(len(extra) // 2))
    else:
        chosen = members
    plt = _plt()
    if axes is None:
        _, axes = plt.subplots(1, len(chosen), figsize=(4.6 * len(chosen), 0.5 * max_display + 1.8), squeeze=False)
        axes = axes[0]
    table = explanation.configurations
    for ax, idx in zip(axes, chosen):
        e = explanation.at(idx)
        if instance is None and e.n_explain > 1:
            bar(e, max_display=max_display, ax=ax)
        else:
            interval_bars(e, instance=0 if instance is None else instance, max_display=max_display, ax=ax, compact=True)
        row = table.loc[idx]
        tag = " (knee)" if idx == explanation.selection.knee else ""
        tag += " (fidelity optimal)" if idx == explanation.fidelity_optimal else ""
        ax.set_title(f"K={int(row['n_concepts'])}, retained {len(e.retained)}: fidelity loss {row['fidelity_loss']:.3f}, instability {row['instability']:.3f}{tag}", fontsize=8.5)
    if show:
        plt.show()
    return axes


def selection_membership(explanation, ax=None, show: bool = False, features: bool = True):
    """How often the Pareto front retains every concept, and every feature, across all its members.

    The selection membership of a concept is the share of the distinct front
    explanations that retain it (Section 3.5); it measures the robustness of
    a concept to the choice of the configuration, which a single explanation
    cannot show. With ``features`` the feature level membership is drawn
    beside it.
    """
    plt = _plt()
    if ax is None:
        _, axs = plt.subplots(1, 2 if features else 1, figsize=(10.5 if features else 5.5, 0.4 * max(explanation.n_concepts, explanation.partition.n_features if features else 0) + 1.2), squeeze=False)
        axs = axs[0]
    else:
        axs = [ax]
        features = False
    csm = np.asarray(explanation.concept_selection_membership, dtype=float)
    order = np.argsort(-csm, kind="stable")
    y = np.arange(len(order))[::-1]
    axs[0].barh(y, csm[order], color=[POSITIVE if k in set(explanation.retained.tolist()) else GRID for k in order], edgecolor=TEXT, linewidth=0.5, height=0.6)
    for yi, val in zip(y, csm[order]):
        axs[0].text(val, yi, f" {val:.2f}", va="center", fontsize=8, color=MUTED)
    axs[0].set_yticks(y)
    axs[0].set_yticklabels(_tick_labels(explanation, order))
    axs[0].set_xlim(0, 1.15)
    axs[0].set_xlabel("concept selection membership (share of the front retaining the concept)")
    style_axes(axs[0])
    if features:
        fsm = np.asarray(explanation.feature_selection_membership, dtype=float)
        order_f = np.argsort(-fsm, kind="stable")
        yf = np.arange(len(order_f))[::-1]
        dominant = explanation.partition.U.argmax(axis=1)
        axs[1].barh(yf, fsm[order_f], color=[PALETTE[dominant[j] % len(PALETTE)] for j in order_f], edgecolor=TEXT, linewidth=0.4, height=0.6, alpha=0.85)
        axs[1].set_yticks(yf)
        axs[1].set_yticklabels([explanation.feature_names[j] for j in order_f], fontsize=8)
        axs[1].set_xlim(0, 1.15)
        axs[1].set_xlabel("feature selection membership (colour: dominant concept)")
        style_axes(axs[1])
    if show:
        plt.show()
    return axs[0] if not features else axs


def coverage(explanation, replicate: np.ndarray, ax=None, show: bool = False):
    """Coverage of a replicated attribution by the support and the core of the fuzzy numbers, per concept.

    ``replicate`` is an independent attribution of the same instances, at the
    feature level (shape (n_explain, n_features), aggregated by membership)
    or at the concept level (shape (n_explain, K)), for example the
    attribution of a model refitted on other data. The nominal levels of the
    support and the core are drawn as dotted lines; bars above them mean the
    intervals are conservative, below them that they are too narrow.
    """
    R = np.asarray(replicate, dtype=float)
    U = explanation.partition.U
    if R.shape == (explanation.n_explain, U.shape[0]):
        R = R @ U
    elif R.shape != (explanation.n_explain, explanation.n_concepts):
        raise ValueError(f"replicate must have shape ({explanation.n_explain}, {U.shape[0]}) or ({explanation.n_explain}, {explanation.n_concepts})")
    Q = explanation.quantiles
    sup = ((R >= Q[..., 0]) & (R <= Q[..., 4])).mean(axis=0)
    core = ((R >= Q[..., 1]) & (R <= Q[..., 3])).mean(axis=0)
    nominal_sup = (explanation.support_percentiles[1] - explanation.support_percentiles[0]) / 100.0
    nominal_core = (explanation.core_percentiles[1] - explanation.core_percentiles[0]) / 100.0
    plt, ax = _axes(ax, (6.5, 0.5 * explanation.n_concepts + 1.4))
    y = np.arange(explanation.n_concepts)[::-1]
    ax.barh(y + 0.18, sup, height=0.34, color=POSITIVE, alpha=0.5, label=f"support (nominal {nominal_sup:.2f})")
    ax.barh(y - 0.18, core, height=0.34, color=POSITIVE, label=f"core (nominal {nominal_core:.2f})")
    ax.axvline(nominal_sup, color=POSITIVE, linestyle=":", linewidth=1.2)
    ax.axvline(nominal_core, color=TEXT, linestyle=":", linewidth=1.2)
    for yi, s_, c_ in zip(y, sup, core):
        ax.text(s_, yi + 0.18, f" {s_:.2f}", va="center", fontsize=7.5, color=MUTED)
        ax.text(c_, yi - 0.18, f" {c_:.2f}", va="center", fontsize=7.5, color=MUTED)
    ax.set_yticks(y)
    ax.set_yticklabels(_tick_labels(explanation, np.arange(explanation.n_concepts)))
    ax.set_xlim(0, 1.12)
    ax.set_xlabel("share of instances whose replicated attribution falls inside the interval")
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    return _finish(plt, ax, show)
