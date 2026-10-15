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

The views that only SPARC can draw live in :mod:`sparc.plotting` and are
re-exported here (``fuzzy_attribution``, ``linguistic_bars``, ``pareto_front``,
``membership``).
"""

from __future__ import annotations

import numpy as np

from sparc.plotting import MUTED, PALETTE, TEXT, _plt, style_axes
from sparc.plotting import plot_fuzzy_attribution as fuzzy_attribution
from sparc.plotting import plot_linguistic_bars as linguistic_bars
from sparc.plotting import plot_membership as membership
from sparc.plotting import plot_pareto_front as pareto_front

POSITIVE = PALETTE[0]
NEGATIVE = PALETTE[1]

__all__ = ["bar", "waterfall", "beeswarm", "heatmap", "dependence", "force", "decision", "fuzzy_attribution", "linguistic_bars", "pareto_front", "membership"]


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
