#!/usr/bin/env python3
r"""Build every chart the thesis needs from the frozen result set.

Every number below is transcribed from `research/results/RESULTS.md` or read directly
from the artifacts under `research/results/`; nothing is estimated or smoothed. Run from
this directory:

    python3 make_figures.py

Outputs one PDF (for \\includegraphics under pdflatex) and one PNG (for quick viewing)
per figure, into this directory.
"""

from __future__ import annotations

import json
import os
import textwrap
from typing import Any

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
RESULTS = os.path.join(REPO, 'research', 'results')
# figures are written next to the LaTeX sources that include them
ASSETS = os.path.abspath(os.path.join(HERE, '..', 'assets'))

# Print-safe palette, distinguishable in greyscale by lightness order.
#
# Six reserved colours, with the same meaning in every figure where the thing they name
# is an axis or a curve:
#
#   scores   SARI and D-SARI -> GOLD          BLEU -> BLUE
#   models   untuned bart-base -> GREY        fine-tuned bart-base -> GREEN
#            prompted LLM (7B) -> VIOLET      prompted LLM (3B) -> PINK
#
# Reserved models also keep one marker and line style, for greyscale and colour-blind
# readers:
#
#   untuned bart-base          grey         square, dashed
#   fine-tuned, full scope     green        filled circle, solid
#   fine-tuned, reduced scope  light green  open circle, dotted
#   prompted LLM (7B)          violet       diamond, dotted
#
# Other series use BROWN, TAN or INK; RED is for thresholds and failures. No teal or
# orange: too close to the reserved green/blue and gold. Axis labels and ticks are black
# unless the axis is SARI/D-SARI or BLEU. fig_labelled_verdicts and the two diagrams
# have no score axis or model curve and keep their own semantics.
INK = '#1a1a1a'
GREY = '#8a8a8a'
LIGHT = '#cfcfcf'
BLUE = '#2f5d8a'
GOLD = '#9a7400'  # dark yellow: a bright one is unreadable as a line or a label
GREEN = '#4a7c59'
LIGHT_GREEN = '#7aab88'
RED = '#9e3b3b'
VIOLET = '#6b4c9a'
PINK = '#b5487f'
BROWN = '#6f4a2f'
TAN = '#c9a37a'

plt.rcParams.update(
    {
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'font.size': 9,
        'font.family': 'serif',
        'axes.edgecolor': INK,
        'axes.labelcolor': INK,
        'axes.titlesize': 10,
        'axes.titleweight': 'bold',
        'axes.spines.top': False,
        'axes.spines.right': False,
        'xtick.color': INK,
        'ytick.color': INK,
        'text.color': INK,
        'legend.frameon': False,
        'grid.color': LIGHT,
        'grid.linewidth': 0.6,
    }
)


def save(fig, name: str) -> None:
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(ASSETS, f'{name}.{ext}'), bbox_inches='tight')
    plt.close(fig)
    print(f'wrote {name}.pdf / .png')


# Figure 1 — S6: BLEU runs inverse to SARI, and the edit rate is the mechanism
# (RESULTS.md §1, row S6).
def fig_bleu_sari_inversion() -> None:
    # Plotted order (untuned, comparison, fine-tuned, 7B, 3B) is strictly descending
    # BLEU, which makes the rising SARI curve read as the inversion. Checkpoint names
    # are in the caption, not on the axis.
    #   (label in (a), label in (b), SARI, BLEU, % of outputs identical to the input)
    conds = [
        ('untuned model', 'untuned', 21.39, 91.64, 93.9),
        ('comparison simplifier', 'comparison', 38.17, 88.58, 13.4),
        ('fine-tuned model', 'fine-tuned', 37.80, 88.30, 11.7),
        ('prompted 7B LLM', '7B LLM', 45.95, 75.46, 4.5),
        ('prompted 3B LLM', '3B LLM', 47.03, 68.21, 1.2),
    ]
    labels = [c[0] for c in conds]
    sari = [c[2] for c in conds]
    bleu = [c[3] for c in conds]
    x = list(range(len(conds)))

    fig = plt.figure(figsize=(7.4, 3.0))
    outer = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.0], wspace=0.62)

    # (a) One axis per metric: they share a nominal 0--100 range but occupy disjoint
    # parts of it. Only the *direction* of the two curves is comparable.
    ax1 = fig.add_subplot(outer[0])
    ax1b = ax1.twinx()
    ax1.plot(x, sari, 'o-', color=GOLD, lw=1.6, ms=5)
    ax1b.plot(x, bleu, 's-', color=BLUE, lw=1.6, ms=5)

    ax1.set_xticks(x)
    ax1.set_xticklabels(
        labels, fontsize=7.5, rotation=45, ha='right', rotation_mode='anchor'
    )
    ax1.set_ylabel('SARI', color=GOLD)
    ax1.set_ylim(15, 52)
    ax1.set_yticks([20, 30, 40, 50])
    # colour tick values and axis label only, not spine or tick marks
    ax1.tick_params(axis='y', labelcolor=GOLD)
    ax1.yaxis.grid(True)
    ax1.set_axisbelow(True)

    ax1b.set_ylabel('BLEU', color=BLUE, rotation=270, va='bottom')
    ax1b.set_ylim(62, 96)
    ax1b.set_yticks([70, 80, 90])
    ax1b.tick_params(axis='y', labelcolor=BLUE)
    ax1b.spines['right'].set_visible(True)
    ax1b.spines['top'].set_visible(False)
    ax1.set_title('(a) Conditions ordered by BLEU')

    # (b) Two broken axes cutting out the empty ranges: SARI 23--35 and 16--89%
    # unchanged. The SARI cut can't start at 20 (untuned sits at 21.39); the other ends
    # at 89 so the 90 tick clears the break mark. All four cells keep one units-per-inch
    # scale per axis.
    # Label offsets sit between gridlines: the 5% and 10% lines fall where the default
    # offsets put "7B LLM" and "fine-tuned".
    offsets = {
        'untuned': ((8, -1), 'left'),
        'fine-tuned': ((8, -3), 'left'),
        'comparison': ((8, 3), 'left'),
        '7B LLM': ((-5, 8), 'right'),
        '3B LLM': ((6, 1), 'left'),
    }
    pts = [(c[1], c[2], c[4]) + offsets[c[1]] for c in conds]
    xlims = [(19.0, 23.0), (35.0, 51.0)]
    ylims = [(89.0, 100.0), (-2.0, 16.0)]
    yticks = [[90, 95, 100], [0, 5, 10, 15]]
    xticks = [[20, 22], [36, 40, 44, 48]]

    host = fig.add_subplot(outer[1], frameon=False)
    host.set_xticks([])
    host.set_yticks([])
    host.set_title('(b) Higher SARI, fewer unedited outputs')
    host.set_xlabel('SARI', labelpad=20, color=GOLD)
    host.set_ylabel('outputs identical to input (%)', labelpad=26)

    inner = outer[1].subgridspec(
        2,
        2,
        width_ratios=[xlims[0][1] - xlims[0][0], xlims[1][1] - xlims[1][0]],
        height_ratios=[ylims[0][1] - ylims[0][0], ylims[1][1] - ylims[1][0]],
        wspace=0.10,
        hspace=0.14,
    )
    cell = {}
    for r, ylim in enumerate(ylims):
        for c, xlim in enumerate(xlims):
            a = fig.add_subplot(inner[r, c])
            # transparent, so a point label may run into the neighbouring empty cell
            a.patch.set_visible(False)
            a.set_xlim(*xlim)
            a.set_ylim(*ylim)
            a.set_xticks(xticks[c])
            a.set_yticks(yticks[r])
            a.tick_params(labelsize=7.5)
            a.yaxis.grid(True)
            a.set_axisbelow(True)
            for name, sx, sy, off, ha in pts:
                if xlim[0] <= sx <= xlim[1] and ylim[0] <= sy <= ylim[1]:
                    a.plot(sx, sy, 'o', color=INK, ms=6)
                    a.annotate(
                        name,
                        (sx, sy),
                        textcoords='offset points',
                        xytext=off,
                        fontsize=7.5,
                        ha=ha,
                        va='center',
                        clip_on=False,
                        annotation_clip=False,
                    )
            cell[(r, c)] = a

    top_l, top_r, bot_l, bot_r = cell[(0, 0)], cell[(0, 1)], cell[(1, 0)], cell[(1, 1)]
    for a in (top_l, top_r):  # the row above the y-axis break
        a.spines['bottom'].set_visible(False)
        a.tick_params(axis='x', bottom=False, labelbottom=False)
    for a in (top_r, bot_r):  # the column right of the x-axis break
        a.spines['left'].set_visible(False)
        a.tick_params(axis='y', left=False, labelleft=False)
    for a in (bot_l, bot_r):  # SARI values in gold, as in (a)
        a.tick_params(axis='x', labelcolor=GOLD, labelsize=7.5)

    # Break marks only on the broken axes: y on the left spine, x on the bottom spine.
    mark = dict(
        marker=[(-1, -0.6), (1, 0.6)],
        markersize=5,
        linestyle='none',
        color=INK,
        mec=INK,
        mew=1.0,
        clip_on=False,
    )
    top_l.plot([0], [0], transform=top_l.transAxes, **mark)
    bot_l.plot([0], [1], transform=bot_l.transAxes, **mark)
    bot_l.plot([1], [0], transform=bot_l.transAxes, **mark)
    bot_r.plot([0], [0], transform=bot_r.transAxes, **mark)

    save(fig, 'fig_bleu_sari_inversion')


# Figure 2 — the BLEU reference-count sweep (RESULTS.md §1, k-sweep table)
def fig_bleu_ksweep() -> None:
    k = [1, 2, 3, 5, 9, 10]
    baseline = [43.70, 59.86, 68.51, 81.82, 90.43, 91.64]
    finetuned = [40.15, 57.10, 66.38, 78.54, 87.52, 88.30]

    fig, ax = plt.subplots(figsize=(4.0, 2.9))
    ax.plot(k, baseline, 's--', color=GREY, lw=1.6, ms=5, label='untuned bart-base')
    ax.plot(k, finetuned, 'o-', color=GREEN, lw=1.6, ms=5, label='fine-tuned bart-base')
    for kk, b, f in zip(k, baseline, finetuned, strict=True):
        ax.plot([kk, kk], [f, b], color=LIGHT, lw=1.0, zorder=0)
    ax.set_xlabel('references per sentence ($k$)')
    ax.set_ylabel('BLEU', color=BLUE)
    ax.tick_params(axis='y', labelcolor=BLUE)
    ax.set_xticks(k)
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)
    ax.legend(loc='lower right', fontsize=8, handlelength=3.2)
    save(fig, 'fig_bleu_ksweep')


# Figure 3 — D-SARI and LENS rank the same three systems in opposite orders
# (RESULTS.md §2, rows D6 / D6').
def fig_metric_disagreement() -> None:
    systems = ['untuned', 'fine-tuned', 'prompted LLM (7B)']
    dsari = [14.42, 34.93, 22.39]
    lens = [33.72, 45.43, 69.22]
    colours = [GREY, GREEN, VIOLET]
    marks = ['s--', 'o-', 'D:']  # the reserved marker and line style per model

    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    for name, dsari_v, lens_v, c, m in zip(
        systems, dsari, lens, colours, marks, strict=True
    ):
        ax.plot([0, 1], [dsari_v, lens_v], m, color=c, lw=1.8, ms=6, label=name)
        ax.annotate(
            f'{dsari_v:.2f}',
            (0, dsari_v),
            textcoords='offset points',
            xytext=(-8, -3),
            ha='right',
            fontsize=8,
            color=c,
        )
        ax.annotate(
            f'{lens_v:.2f}',
            (1, lens_v),
            textcoords='offset points',
            xytext=(8, -3),
            ha='left',
            fontsize=8,
            color=c,
        )
    ax.set_xlim(-0.45, 1.45)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(
        ['D-SARI\n(n-gram, single reference)', 'LENS\n(learned, human ratings)']
    )
    ax.set_ylabel('score')
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)
    # handlelength 3.2: at the default 2.0, dotted and dashed keys are hard to tell apart
    ax.legend(
        loc='upper center',
        bbox_to_anchor=(0.5, -0.22),
        fontsize=7.5,
        ncol=3,
        columnspacing=1.2,
        handletextpad=0.5,
        handlelength=3.2,
    )
    save(fig, 'fig_metric_disagreement')


# Figure 4 — compression as a policy, not a fixed output length
# Read from the two qualitative-review artifacts (Q1' and Q2).
def fig_compression_by_length() -> None:
    q1p = json.load(
        open(
            os.path.join(
                RESULTS,
                'remote',
                'normaliser_isolation',
                'qualitative_review_old_ckpt.json',
            )
        )
    )
    q2 = json.load(
        open(
            os.path.join(
                RESULTS, 'remote', 'document_eval_full', 'qualitative_review_full.json'
            )
        )
    )

    buckets = [
        b['source_words'] for b in q2['compression_by_source_length']['finetuned']
    ]
    reduced = [
        b['prediction_length_ratio_median']
        for b in q1p['compression_by_source_length']['finetuned']
    ]
    full = [
        b['prediction_length_ratio_median']
        for b in q2['compression_by_source_length']['finetuned']
    ]
    human = [
        b['reference_length_ratio_median']
        for b in q2['compression_by_source_length']['finetuned']
    ]
    zero = [
        b['prediction_length_ratio_median']
        for b in q2['compression_by_source_length']['baseline']
    ]
    x = range(len(buckets))

    fig, ax = plt.subplots(figsize=(4.6, 3.1))
    # Human reference is not a reserved model, so ink. Both checkpoints are fine-tuned
    # bart-base; the reduced-scope one (same model, trained on less) is a lighter, open,
    # dotted variant of the full-scope green, not a different colour.
    ax.plot(x, zero, 's--', color=GREY, lw=1.5, ms=6, label='untuned bart-base')
    ax.plot(x, human, '^-', color=INK, lw=1.6, ms=6, label='human reference')
    ax.plot(
        x,
        full,
        'o-',
        color=GREEN,
        lw=1.8,
        ms=6,
        label='full-scope checkpoint (n=8,000)',
    )
    ax.plot(
        x,
        reduced,
        'o:',
        color=LIGHT_GREEN,
        lw=1.6,
        ms=6,
        mfc='white',
        mec=LIGHT_GREEN,
        mew=1.4,
        label='reduced-scope checkpoint (n=500)',
    )
    ax.set_xticks(list(x))
    ax.set_xticklabels([b.replace('-+', '+').replace('-', '–') for b in buckets])
    ax.set_xlabel('source length (words)')
    ax.set_ylabel('median output / source length ratio')
    ax.set_ylim(0, 1.25)
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)
    ax.legend(fontsize=8, loc='lower left', handlelength=3.2)
    save(fig, 'fig_compression_by_length')


# Figure 5 — coverage is a property of the page category
# Read from the site-audit artifact (RESULTS.md §2g).
def fig_skip_share() -> None:
    audit = json.load(
        open(os.path.join(RESULTS, 'site_audit', 'site_audit_20260826T044008Z.json'))
    )
    per = {}
    for row in audit['rows']:
        per.setdefault(row['site']['name'], {'category': row['site']['category']})
        per[row['site']['name']][row['cut']] = row['coverage']['skipped_share'] * 100

    order = sorted(per.items(), key=lambda kv: (kv[1]['category'], kv[0]))
    names = [k for k, _ in order]
    cats = [v['category'] for _, v in order]
    sent = [v['sentence'] for _, v in order]
    doc = [v['document'] for _, v in order]

    fig, ax = plt.subplots(figsize=(7.0, 3.3))
    idx = range(len(names))
    w = 0.4
    ax.bar([i - w / 2 for i in idx], sent, width=w, color=BROWN, label='sentence cut')
    ax.bar(
        [i + w / 2 for i in idx],
        doc,
        width=w,
        color=TAN,
        edgecolor=BROWN,
        lw=0.5,
        label='document cut',
    )
    ax.set_xticks(list(idx))
    # two lines at 45 degrees: twelve page names don't fit on one row
    ax.set_xticklabels(
        [f'{n}\n({c})' for n, c in zip(names, cats, strict=True)],
        fontsize=7,
        rotation=45,
        ha='right',
        rotation_mode='anchor',
    )
    ax.set_xlim(-0.7, len(names) - 0.3)
    ax.set_ylabel('candidates skipped (%)')
    ax.set_ylim(0, 100)
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)
    # outside the plotting area: at these bar heights every in-axes corner is occupied
    ax.legend(fontsize=8, loc='lower left', bbox_to_anchor=(0.0, 1.01), ncol=2)
    save(fig, 'fig_skip_share')


# Figure 6 — the hand-labelled verdicts (RESULTS.md §2g)
def fig_labelled_verdicts() -> None:
    fig, ax = plt.subplots(figsize=(5.4, 2.0))
    rows = [
        (
            'skipped\n(n=69)',
            [
                ('correct', 48, GREEN),
                ('should have been sent', 15, RED),
                ('correct but costly', 6, GREY),
            ],
        ),
        (
            'unchanged\n(n=61)',
            [
                ('already plain', 49, GREEN),
                ('should have been simplified', 9, RED),
                ('language unsupported', 3, GREY),
            ],
        ),
    ]
    # Bars are shares of each row's own sample (so rows are comparable); segment labels
    # are the raw counts.
    for y, (_label, parts) in enumerate(rows):
        total = sum(n for _, n, _ in parts)
        left = 0.0
        for _name, n, colour in parts:
            share = n / total * 100
            ax.barh(y, share, left=left, color=colour, edgecolor='white', height=0.55)
            ax.text(
                left + share / 2,
                y,
                str(n),
                va='center',
                ha='center',
                color='white',
                fontsize=8,
            )
            left += share
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows], multialignment='right')
    ax.set_xlabel("share of the row's hand-labelled items (%)")
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.invert_yaxis()
    ax.xaxis.grid(True)
    ax.set_axisbelow(True)
    handles = [
        Line2D([], [], color=GREEN, lw=7, label='correct behaviour'),
        Line2D([], [], color=RED, lw=7, label='false negative'),
        Line2D([], [], color=GREY, lw=7, label='correct but costly / out of scope'),
    ]
    ax.legend(
        handles=handles,
        fontsize=7.5,
        loc='upper center',
        bbox_to_anchor=(0.5, -0.42),
        ncol=3,
    )
    save(fig, 'fig_labelled_verdicts')


# Figure 7 — the training-set document length distribution against the 512-token ceiling
# Order statistics from RESULTS-backed Finding 6 (n=1,500 source documents, BART tokenizer).
def fig_doclen_distribution() -> None:
    quantiles = [
        (0.50, 120),
        (0.68, 200),
        (0.75, 254),
        (0.92, 450),
        (0.95, 510),
        (1.00, 1168),
    ]
    xs = [q[1] for q in quantiles]
    ys = [q[0] * 100 for q in quantiles]

    fig, ax = plt.subplots(figsize=(4.4, 2.8))
    # Monotone (PCHIP) interpolation through the six measured quantiles (a step plot
    # read as six sharp edges). Markers are the measurements; the line is a smooth
    # join, not data, and the caption says so.
    from scipy.interpolate import PchipInterpolator

    span = xs[-1] - xs[0]
    grid = [xs[0] + span * i / 400.0 for i in range(401)]
    ax.plot(grid, PchipInterpolator(xs, ys)(grid), '-', color=INK, lw=1.8)
    ax.plot(xs, ys, 'o', color=INK, ms=5)
    # 450 and 510 are 60 tokens and 3 points apart: one label above-left, one
    # below-right. 120 goes left of the curve, clear of the mean rule at 180.
    label_off = [
        (-4, 3, 'right'),
        (4, -10, 'left'),
        (4, -10, 'left'),
        (-4, 4, 'right'),
        (5, -11, 'left'),
        (4, -10, 'left'),
    ]
    for (x, y), (dx, dy, ha) in zip(zip(xs, ys, strict=True), label_off, strict=True):
        ax.annotate(
            f'{x}',
            (x, y),
            textcoords='offset points',
            xytext=(dx, dy),
            fontsize=7.5,
            ha=ha,
        )
    ax.axvline(512, color=RED, lw=1.4, ls='--')
    ax.annotate(
        'max_length = 512',
        (512, 46),
        textcoords='offset points',
        xytext=(6, 0),
        fontsize=8,
        color=RED,
    )
    ax.axvline(180, color=GREY, lw=1.0, ls=':')
    # level with the max_length label
    ax.annotate(
        'mean 180',
        (180, 46),
        textcoords='offset points',
        xytext=(6, 0),
        fontsize=7.5,
        color=GREY,
    )
    ax.set_xlabel('source document length (BART tokens)')
    ax.set_ylabel('documents at or below (%)')
    ax.set_xlim(0, 1200)
    ax.set_ylim(40, 103)
    ax.set_yticks([40, 50, 60, 70, 80, 90, 100])
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)
    save(fig, 'fig_doclen_distribution')


# Figure 8 — latency by request regime, and what contention does to each
# RESULTS.md §2c, rows P1–P4 and the contention table.
@plt.rc_context({'mathtext.fontset': 'dejavuserif'})
def fig_latency_regimes() -> None:
    # Three panels: per-request seconds (1--21), per-page seconds (15--133) and the
    # contention pair don't share a scale. Horizontal bars so names read straight;
    # every bar is annotated.
    conds = [
        'fine-tuned\n(sentence)',
        'comparison\n(sentence)',
        'fine-tuned\n(document)',
        'prompted 3B\n(sentence)',
    ]
    median = [1.26, 3.07, 19.13, 13.08]
    p95 = [3.28, 11.07, 19.14, 20.45]
    wall = [15.5, 43.6, 22.5, 132.7]

    fig = plt.figure(figsize=(7.4, 3.3))
    outer = fig.add_gridspec(1, 2, width_ratios=[2.05, 1.0], wspace=0.46)
    left = outer[0].subgridspec(1, 2, width_ratios=[1.3, 0.9], wspace=0.09)
    ax1 = fig.add_subplot(left[0])
    ax2 = fig.add_subplot(left[1], sharey=ax1)
    ax3 = fig.add_subplot(outer[1])

    def label_ends(ax, ys, vals, pad, fmt='{:.1f}'):
        for y, v in zip(ys, vals, strict=True):
            ax.text(v + pad, y, fmt.format(v), va='center', ha='left', fontsize=7)

    y = list(range(len(conds)))
    h = 0.36
    ax1.barh(
        [i - h / 2 for i in y], median, height=h, color=BROWN, label='median request'
    )
    ax1.barh(
        [i + h / 2 for i in y],
        p95,
        height=h,
        color=TAN,
        edgecolor=BROWN,
        lw=0.5,
        label='$P_{95}$ request',
    )
    label_ends(ax1, [i - h / 2 for i in y], median, 0.4)
    label_ends(ax1, [i + h / 2 for i in y], p95, 0.4)
    ax1.set_yticks(y)
    ax1.set_yticklabels(conds, fontsize=7.5, multialignment='right')
    ax1.invert_yaxis()
    ax1.set_xlim(0, 26)
    ax1.set_xticks([0, 5, 10, 15, 20, 25])
    ax1.set_xlabel('seconds per request')
    ax1.xaxis.grid(True)
    ax1.set_axisbelow(True)
    ax1.legend(fontsize=7.5, loc='upper right')
    ax1.set_title('(a) Per request', loc='left')

    ax2.barh(y, wall, height=h * 1.6, color=BROWN)
    label_ends(ax2, y, wall, 3.0)
    ax2.tick_params(axis='y', left=False, labelleft=False)
    ax2.set_xlim(0, 168)
    ax2.set_xticks([0, 50, 100, 150])
    ax2.set_xlabel('seconds per page')
    ax2.xaxis.grid(True)
    ax2.set_axisbelow(True)
    ax2.set_title('(b) Whole page', loc='left')

    labels = ['sentence\nwall clock', 'sentence\n$P_{95}$', 'document\nwall clock']
    idle = [15.5, 3.28, 22.5]
    loaded = [121.5, 62.7, 21.8]
    y2 = list(range(len(labels)))
    # Same rule as (a): dark = baseline case, light = stressed (p95 there, loaded
    # machine here). Red read as a third encoding next to the brown.
    ax3.barh(
        [i - h / 2 for i in y2], idle, height=h, color=BROWN, label='idle run'
    )
    ax3.barh(
        [i + h / 2 for i in y2],
        loaded,
        height=h,
        color=TAN,
        edgecolor=BROWN,
        lw=0.5,
        label='busy machine',
    )
    for i, (a, b) in enumerate(zip(idle, loaded, strict=True)):
        ax3.text(
            max(a, b) + 3.0,
            i,
            f'{b / a:.1f}$\\times$',
            va='center',
            ha='left',
            fontsize=8,
        )
    ax3.set_yticks(y2)
    ax3.set_yticklabels(labels, fontsize=7.5, multialignment='right')
    ax3.invert_yaxis()
    ax3.set_xlim(0, 168)
    ax3.set_xticks([0, 50, 100, 150])
    ax3.set_xlabel('seconds')
    ax3.xaxis.grid(True)
    ax3.set_axisbelow(True)
    ax3.legend(fontsize=7.5, loc='lower right')
    ax3.set_title('(c) Under CPU contention', loc='left')

    save(fig, 'fig_latency_regimes')


# Figure 9 — validation loss is a poor proxy for D-SARI
# Loss curve read from the full-scope document run's training history.
def fig_loss_vs_dsari() -> None:
    hist = json.load(
        open(os.path.join(RESULTS, 'remote', 'training_history_document_full.json'))
    )
    evals = [(h['epoch'], h['eval_loss']) for h in hist if 'eval_loss' in h]
    ep = [e for e, _ in evals]
    loss = [v for _, v in evals]

    fig, ax = plt.subplots(figsize=(4.6, 2.9))
    # validation loss is not one of the reserved scores: unreserved curve, black axis
    ax.plot(ep, loss, '-', color=INK, lw=1.6)
    ax.plot(ep, loss, 'o', color=INK, ms=3)
    ax.set_xlabel('epoch')
    ax.set_ylabel('validation loss')
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)
    ax.annotate(
        'epoch 2: 0.3408',
        (2.0, 0.3408),
        textcoords='offset points',
        xytext=(6, 10),
        fontsize=7.5,
        color=INK,
    )
    # "since epoch 2": the 2.0% is relative to the other annotated point
    ax.annotate(
        'epoch 5: 0.3341\n($-$2.0% since epoch 2)',
        (5.0, 0.3341),
        textcoords='offset points',
        xytext=(-78, 12),
        fontsize=7.5,
        color=INK,
    )

    ax2 = ax.twinx()
    ax2.spines['right'].set_visible(True)
    ax2.plot([2.0, 4.0], [31.62, 35.44], 's-', color=GOLD, lw=1.6, ms=6)
    ax2.set_ylabel('document D-SARI', color=GOLD)
    ax2.tick_params(axis='y', labelcolor=GOLD)
    ax2.set_ylim(28, 38)
    # above the segment rather than on it: at x=3 the line is at 33.53
    ax2.annotate('+3.82 D-SARI', (3.0, 34.85), fontsize=8, color=GOLD, ha='center')
    save(fig, 'fig_loss_vs_dsari')


def _box(
    ax,
    x,
    y,
    w,
    h,
    title,
    sub=None,
    fc='white',
    ec=INK,
    fs=8.5,
    title_dy=None,
    sub_dy=None,
    sub_ls=1.6,
    tc=INK,
    sub_tc='#4a4a4a',
):
    """One component. The default proportional text placement only reads correctly at
    one box height; `title_dy` and `sub_dy` (each a distance below the box's top edge)
    and `sub_ls` (line spacing) place the text explicitly, which a box several times
    taller than its own text needs.
    """
    from matplotlib.patches import FancyBboxPatch

    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle='round,pad=0.0,rounding_size=0.08',
            linewidth=1.0,
            edgecolor=ec,
            facecolor=fc,
            zorder=2,
        )
    )
    if sub:
        ty = y + h * 0.62 if title_dy is None else y + h - title_dy
        sy = y + h * 0.26 if sub_dy is None else y + h - sub_dy
        ax.text(
            x + w / 2,
            ty,
            title,
            ha='center',
            va='center',
            fontsize=fs,
            color=tc,
            zorder=3,
        )
        ax.text(
            x + w / 2,
            sy,
            sub,
            ha='center',
            va='center',
            fontsize=fs - 2.0,
            color=sub_tc,
            zorder=3,
            linespacing=sub_ls,
        )
    else:
        ax.text(
            x + w / 2,
            y + h / 2,
            title,
            ha='center',
            va='center',
            fontsize=fs,
            color=tc,
            zorder=3,
        )


def _band(ax, x, y, w, h, label, fc='#f1f1f1', ls=(0, (4, 3)), fs=8, z=1.0):
    """One tier of the architecture figure. `z` orders nested bands: a sub-area needs a
    higher zorder than the band enclosing it, and both stay below the boxes (zorder 2).
    """
    from matplotlib.patches import FancyBboxPatch

    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle='round,pad=0.0,rounding_size=0.12',
            linewidth=0.8,
            edgecolor=GREY,
            facecolor=fc,
            linestyle=ls,
            zorder=z,
        )
    )
    ax.text(
        x + 0.12,
        y + h - 0.26,
        label,
        ha='left',
        va='center',
        fontsize=fs,
        style='italic',
        color='#4a4a4a',
        zorder=z + 0.1,
    )


def _arrow(
    ax,
    xy_from,
    xy_to,
    label=None,
    rad=0.0,
    fs=7.5,
    label_offset=(0, 0.12),
    shrinkA=1,
    shrinkB=1,
    label_colour='#4a4a4a',
    label_ls=1.15,
    label_va='center',
    colour=INK,
):
    """`shrinkA=0` for a leg that continues a `_path`: the default 1pt back-off leaves
    a visible gap at the corner where the two meet.
    """
    ax.annotate(
        '',
        xy=xy_to,
        xytext=xy_from,
        arrowprops=dict(
            arrowstyle='-|>',
            color=colour,
            lw=1.1,
            connectionstyle=f'arc3,rad={rad}',
            shrinkA=shrinkA,
            shrinkB=shrinkB,
        ),
        zorder=4,
    )
    if label:
        mx = (xy_from[0] + xy_to[0]) / 2 + label_offset[0]
        my = (xy_from[1] + xy_to[1]) / 2 + label_offset[1]
        # `label_va="baseline"` for labels that must sit on one optical line with a
        # neighbour: centring includes the descenders, so "reads" and
        # "runtime.sendMessage" end up a hair apart under va="center".
        ax.text(
            mx,
            my,
            label,
            ha='center',
            va=label_va,
            fontsize=fs,
            color=label_colour,
            linespacing=label_ls,
            zorder=5,
        )


def _path(ax, pts, colour=INK):
    """A multi-segment connector drawn without an arrowhead, for a leg that merges into
    another path instead of terminating at a box.
    """
    ax.plot(
        [x for x, _ in pts],
        [y for _, y in pts],
        color=colour,
        lw=1.1,
        solid_capstyle='butt',
        solid_joinstyle='round',
        zorder=4,
    )


# Figure 10 — system architecture
# matplotlib rather than TikZ: the paper builds with graphicx alone and one command
# regenerates every figure.
def fig_architecture() -> None:
    # Vertical tiers with serpentine rows (browser left-to-right, FastAPI right-to-left),
    # so extract_and_clean sits directly beneath the service worker and the request is
    # a straight drop. Down is the request, up is the response.
    #
    # Both responses share one corridor clear of every band: a cache hit *is* a
    # previously guarded response (the cache stores the post-guard verdict; main.py's
    # _batch_worker writes it), so two separate returns would imply a distinction the
    # backend does not make.
    #
    # Box widths are measured text extent plus ~0.20 padding a side ("per-model batch
    # queue" alone needs 3.11 of a sub-area's 10.30). Re-measure after editing any label.
    fig, ax = plt.subplots(figsize=(7.2, 5.80))
    ax.set_xlim(0, 12)
    ax.set_ylim(-0.02, 9.65)
    ax.axis('off')

    # sub-areas: lighter fill, dotted rule, smaller label, so they don't read as a tier
    SUB: dict[str, Any] = dict(fc='#fafafa', ls=(0, (1, 2.5)), fs=7.5, z=1.5)

    # browser tier: band 0.41 taller than the boxes need, so the four API labels above
    # and below them don't crowd its edges (they did at 2.09)
    _band(ax, 0.20, 7.05, 11.65, 2.50, 'browser — Chrome MV3 extension')
    _box(ax, 0.45, 7.62, 1.40, 0.95, 'page\nDOM')
    _box(
        ax,
        2.50,
        7.62,
        3.70,
        0.95,
        'content script',
        'selection · chunking · write-back',
    )
    _box(
        ax,
        6.80,
        7.62,
        4.30,
        0.95,
        'service worker',
        'model key · 8-slot queue · status check',
    )
    # "reads"/"writes": both verbs take the content script as subject ("receives" would
    # not). Labels above/below the boxes, as for the messaging pair: the 0.65 gap is too
    # narrow for a word.
    _arrow(
        ax,
        (1.85, 8.29),
        (2.50, 8.29),
        'reads',
        label_offset=(0, 0.42),
        fs=7,
        label_va='baseline',
    )
    _arrow(
        ax,
        (2.50, 7.90),
        (1.85, 7.90),
        'writes',
        label_offset=(0, -0.52),
        fs=7,
        label_va='baseline',
    )
    # Two directions, two Chrome APIs: the content script calls runtime.sendMessage,
    # the worker answers with sendResponse (worker-initiated commands use
    # tabs.sendMessage).
    _arrow(
        ax,
        (6.20, 8.29),
        (6.80, 8.29),
        'runtime.sendMessage',
        label_offset=(0, 0.42),
        fs=7,
        label_va='baseline',
    )
    _arrow(
        ax,
        (6.80, 7.90),
        (6.20, 7.90),
        'sendResponse',
        label_offset=(0, -0.52),
        fs=7,
        label_va='baseline',
    )

    # local backend: two sub-areas, guard stack outside both
    _band(ax, 0.20, 0.12, 11.65, 6.68, 'local backend')
    _band(ax, 0.35, 4.48, 10.65, 1.79, 'FastAPI service — 127.0.0.1', **SUB)
    _band(ax, 0.35, 0.28, 6.30, 3.90, 'models — seven selectable keys', **SUB)

    _box(ax, 8.00, 4.86, 2.80, 0.85, 'extract_and_clean()', 'strips non-text')
    _box(ax, 4.85, 4.86, 2.80, 0.85, 'LRU cache', 'checks earlier outputs')
    _box(ax, 0.55, 4.86, 3.45, 0.85, 'per-model batch queue', '8 items or 0.1 s')

    # The three models this work produced or prompted, a gap, then the published
    # comparison checkpoint: the gap is the distinction.
    # 0.64-high box: title, then subtitle
    MB: dict[str, Any] = dict(title_dy=0.23, sub_dy=0.46)
    _box(
        ax,
        0.70,
        2.96,
        5.60,
        0.64,
        'fine-tuned bart-base',
        'granularities: sentence · document',
        **MB,
    )
    _box(
        ax,
        0.70,
        2.16,
        5.60,
        0.64,
        'prompted LLM (Qwen2.5-7B)',
        'granularities: sentence · document',
        **MB,
    )
    _box(
        ax,
        0.70,
        1.36,
        5.60,
        0.64,
        'prompted LLM (Qwen2.5-3B)',
        'granularities: sentence · document',
        **MB,
    )
    # muted: the one model in this column that this work did not produce or prompt
    _box(
        ax,
        0.70,
        0.42,
        5.60,
        0.64,
        'comparison simplifier',
        'sentence granularity',
        fc='#f6f6f6',
        ec=GREY,
        tc='#787878',
        sub_tc='#8f8f8f',
        **MB,
    )

    # Guard stack beside the models, in neither sub-area: it runs on any model's
    # output, and two of its four checks are method-agnostic. One check per line, in
    # the order _batch_worker applies them (a vertical list implies a sequence).
    # Both boxes use the FastAPI row's title-to-subtitle gap (0.31): tidy() via default
    # placement, the guard stack explicitly (four lines are too tall for the default).
    # The pair is centred on the models sub-area (middle 2.23 = middle of 0.705--3.755).
    _box(ax, 7.25, 2.905, 3.75, 0.85, 'tidy()', 'removes stray spacing')
    _box(
        ax,
        7.25,
        0.705,
        3.75,
        1.85,
        'guard stack',
        'hallucination\nreason codes\ncorpus deny-list\nchange guard',
        title_dy=0.30,
        sub_dy=1.08,
    )

    # request path. Blue for both halves of the HTTP call (line and label): the one hop
    # crossing a process boundary. Label at the top of the 1.9-long drop, below the box
    # it leaves and below sendResponse, still inside the browser band.
    _arrow(ax, (9.60, 7.62), (9.60, 5.71), colour=BLUE)
    ax.text(
        9.45,
        7.22,
        'POST /simplify',
        ha='right',
        va='center',
        fontsize=7,
        color=BLUE,
        zorder=5,
    )
    _arrow(ax, (8.00, 5.285), (7.65, 5.285))
    _arrow(
        ax,
        (4.85, 5.285),
        (4.00, 5.285),
        'cache\nmiss',
        label_offset=(0, 0.32),
        fs=7,
        label_colour=RED,
    )
    _path(ax, [(2.275, 4.86), (2.275, 4.62), (3.50, 4.62)])
    _arrow(ax, (3.50, 4.62), (3.50, 4.21), shrinkA=0)  # into the models sub-area
    # out of the models sub-area at its right middle, up the 0.60 gap between the
    # sub-area and the guard stack, into tidy()'s left edge
    _path(ax, [(6.65, 2.23), (6.95, 2.23), (6.95, 3.33)])
    _arrow(ax, (6.95, 3.33), (7.25, 3.33), shrinkA=0)
    _arrow(ax, (9.125, 2.905), (9.125, 2.555))  # tidy() into the guard stack

    # return corridor. Guarded output: out of the guard stack, up the corridor, into the
    # service worker's right edge. Blue like the drop at 9.60 (same HTTP call), as is
    # the cache-hit leg merging into it (the other way that response is served).
    _path(ax, [(11.00, 1.63), (11.35, 1.63), (11.35, 8.095)], colour=BLUE)
    _arrow(ax, (11.35, 8.095), (11.10, 8.095), shrinkA=0, colour=BLUE)
    # cache hit: down out of the LRU cache, right along the gap between the two
    # sub-areas, into the same corridor
    _path(ax, [(6.25, 4.86), (6.25, 4.33), (11.35, 4.33)], colour=BLUE)
    # left of the leg it labels; one row clears both the sub-area's rule at 4.48 and
    # the horizontal run at 4.33, which two rows could not
    ax.text(
        6.12,
        4.62,
        'cache hit',
        ha='right',
        va='center',
        fontsize=7,
        color=GREEN,
        zorder=5,
    )
    # right of the corridor (hence 11.35, not 11.55): nothing else reaches past it, so
    # the label can sit level with the climb
    ax.text(
        11.62,
        4.85,
        'response: text + reason code',
        rotation=90,
        ha='center',
        va='center',
        fontsize=7,
        color=BLUE,
        zorder=5,
    )

    save(fig, 'fig_architecture')


# Figure 11 — the request path, end to end (draft L280)
def fig_dataflow() -> None:
    # Two columns, one row per step: lanes and arrows on the left, numbered wording on
    # the right (labels on the lifelines made it unclear which arrow they belonged to).
    # lane pitch 1.60 so a 1.42-wide header box fits "backend" clear of its border
    lanes = [
        ('content\nscript', 0.95),
        ('service\nworker', 2.55),
        ('backend', 4.15),
        ('model', 5.75),
    ]
    LANE_W = 1.42
    TEXT_X = 7.00  # left edge of the numbered column
    BADGE_X = 6.55
    WRAP = 51  # characters: the widest line the column has room for
    # 4a/4b, not 4/5: exclusive outcomes of one lookup, not two steps
    NUMS = ['1', '2', '3', '4a', '4b', '5', '6', '7', '8']
    Y_TOP = 8.20  # centre of the first row
    DY = 0.88  # row pitch
    # Arrow plus tag is 0.31 tall, so the line sits 0.155 below the row centre to
    # centre the pair. Self-calls are symmetric and need no offset.
    ARROW_DY = 0.155

    # (from lane, to lane, tag, wording, granularity-dependent, exclusive branch,
    #  tag colour, crosses the HTTP boundary)
    TAG = '#4a4a4a'
    steps = [
        (
            0,
            1,
            'units',
            'Units: one leaf or <br>-chunk (sentence), one heading-delimited '
            'section (document).',
            True,
            False,
            TAG,
            False,
        ),
        (
            1,
            2,
            'POST',
            'POST /simplify with the chosen model key; at most 8 requests in '
            'flight.',
            False,
            False,
            TAG,
            True,
        ),
        (
            2,
            2,
            'clean',
            'extract_and_clean, then a cache lookup.',
            False,
            False,
            TAG,
            False,
        ),
        (
            2,
            1,
            'hit',
            'Cache hit: the stored verdict returns immediately.',
            False,
            True,
            GREEN,
            True,
        ),
        (
            2,
            3,
            'miss',
            'Cache miss: batch queue (8 items or 0.1 s), then generate.',
            False,
            True,
            RED,
            False,
        ),
        (3, 2, 'text', 'The raw generation comes back.', False, False, TAG, False),
        (
            2,
            2,
            'guards',
            'tidy(), then the guard stack, which may restore the original.',
            False,
            False,
            TAG,
            False,
        ),
        (
            2,
            1,
            'reply',
            'Response: input, model_result, simplified, fallback_reason.',
            False,
            False,
            TAG,
            True,
        ),
        (
            1,
            0,
            'write',
            'Write-back across the nodes the unit was built from.',
            True,
            False,
            TAG,
            False,
        ),
    ]

    fig, ax = plt.subplots(figsize=(7.2, 5.05))
    ax.set_xlim(0, 14)
    ax.set_ylim(-0.45, 9.95)
    ax.axis('off')

    y_last = Y_TOP - (len(steps) - 1) * DY

    # alternating row bands tie each arrow to its wording across the whole width
    for i in range(len(steps)):
        if i % 2:
            ax.axhspan(
                Y_TOP - i * DY - DY / 2,
                Y_TOP - i * DY + DY / 2,
                xmin=0.015,
                xmax=1.0,
                facecolor='#f4f4f4',
                zorder=0,
            )

    # Fixed-size boxes, not text bboxes: those size to their text, so two-line names
    # came out taller than "backend" and "model".
    from matplotlib.patches import Ellipse, FancyBboxPatch

    for name, x in lanes:
        ax.add_patch(
            FancyBboxPatch(
                (x - LANE_W / 2, 9.02),
                LANE_W,
                0.82,
                boxstyle='round,pad=0.0,rounding_size=0.08',
                linewidth=0.8,
                edgecolor=GREY,
                facecolor='#f1f1f1',
                zorder=2,
            )
        )
        ax.text(
            x,
            9.43,
            name,
            ha='center',
            va='center',
            fontsize=7.5,
            weight='bold',
            linespacing=1.3,
            zorder=3,
        )
        ax.plot([x, x], [y_last - DY / 2, 8.92], color=LIGHT, lw=1.0, zorder=1)

    for i, (a, b, tag, wording, diverges, branch, tag_colour, api) in enumerate(steps):
        y = Y_TOP - i * DY
        x0, x1 = lanes[a][1], lanes[b][1]
        style = dict(
            arrowstyle='-|>',
            color=BLUE if api else INK,
            lw=1.0,
            linestyle=(0, (3, 2)) if branch else 'solid',
        )
        if a == b:
            # a call the component makes on itself: a small arc to the right of its lane
            ax.annotate(
                '',
                xy=(x0, y - 0.20),
                xytext=(x0, y + 0.20),
                arrowprops=dict(connectionstyle='arc3,rad=-1.3', **style),
                zorder=4,
            )
            ax.text(
                x0 + 0.40,
                y,
                tag,
                ha='left',
                va='center',
                fontsize=7,
                color=tag_colour,
                zorder=4,
            )
        else:
            ya = y - ARROW_DY
            ax.annotate(
                '',
                xy=(x1, ya),
                xytext=(x0, ya),
                arrowprops=dict(connectionstyle='arc3,rad=0.0', **style),
                zorder=4,
            )
            ax.text(
                (x0 + x1) / 2,
                ya + 0.05,
                tag,
                ha='center',
                va='bottom',
                fontsize=7,
                color=tag_colour,
                zorder=4,
            )

        # Step number, violet where the granularities differ. Fixed-size ellipse, not a
        # "circle" bbox (which would make the 4a/4b badges bigger); width and height
        # differ by the axes' unit aspect so it reads as a circle. va="center_baseline"
        # centres the digits; "center" includes descender space and sits them high.
        ax.add_patch(
            Ellipse(
                (BADGE_X, y),
                0.53,
                0.565,
                linewidth=0.9,
                facecolor=VIOLET if diverges else 'white',
                edgecolor=VIOLET if diverges else INK,
                zorder=4,
            )
        )
        ax.text(
            BADGE_X,
            y,
            NUMS[i],
            ha='center',
            va='center_baseline',
            fontsize=7.5,
            zorder=5,
            color='white' if diverges else INK,
        )
        ax.text(
            TEXT_X,
            y,
            textwrap.fill(wording, WRAP),
            ha='left',
            va='center',
            fontsize=7.5,
            linespacing=1.3,
            zorder=4,
        )

    ax.text(
        0.22,
        y_last - 0.72,
        'Steps 1 and 8 are the only two points at which the sentence and document '
        'cuts differ.',
        ha='left',
        va='center',
        fontsize=7.5,
        color=VIOLET,
    )
    ax.text(
        0.22,
        y_last - 1.08,
        'Blue: the HTTP boundary. Step 2 is the API call, steps 4a and 7 its two '
        'possible responses.',
        ha='left',
        va='center',
        fontsize=7.5,
        color=BLUE,
    )
    ax.text(
        0.22,
        y_last - 1.44,
        'Dashed: steps 4a and 4b are the two mutually exclusive outcomes of the '
        'cache lookup.',
        ha='left',
        va='center',
        fontsize=7.5,
        color='#4a4a4a',
    )

    save(fig, 'fig_dataflow')


if __name__ == '__main__':
    fig_bleu_sari_inversion()
    fig_bleu_ksweep()
    fig_metric_disagreement()
    fig_compression_by_length()
    fig_skip_share()
    fig_labelled_verdicts()
    fig_doclen_distribution()
    fig_latency_regimes()
    fig_loss_vs_dsari()
    fig_architecture()
    fig_dataflow()
