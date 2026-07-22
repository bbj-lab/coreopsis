#!/usr/bin/env python3

"""
plot results
"""

import math
import pathlib

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

pd.options.display.float_format = "{:,.3f}".format
pd.options.display.max_columns = None
pd.options.display.max_rows = 100
pd.options.display.width = None
pd.options.display.expand_frame_repr = False
pd.options.display.show_dimensions = True

hm = pathlib.Path("~/Downloads").expanduser().resolve()

# ---------------------------------------------------------------------------
# shared style / configuration (metric-independent)
# ---------------------------------------------------------------------------
# validated categorical palette (light mode)
COL_CURVE = "#737373"  # dark greystone -- site-trained sweep
COL_OTHER = "#789D4A"  # ivy -- fedavg on the other two sites
COL_FED = "#A4343A"  # brick -- fedavg on all three sites
COL_ALL = "#CC8A00"  # dark goldenrod -- single model pooled over all data

# uniform label/mark sizing after rescaling:
# each figure is exported at its own pixel width (multi-panel vs. single panel)
# but rescaled to a common width in the manuscript. Sizing every text and mark
# element proportionally to the figure's own width makes labels, ticks, markers,
# and lines appear uniform once all figures share the same final width.
REF_WIDTH = 900  # width at which the base sizes below apply (scale factor 1.0)
FONT_SIZE = 30  # axis titles, tick labels, legend entries
TITLE_SIZE = 42  # main figure title
SUBTITLE_SIZE = 36  # subplot (panel) titles
LINE_WIDTH = 2
MARKER_SIZE = 8

# dataset -> (display name, fedavg model trained on the *other two* sites,
#             full training-set size)
plot_dsets = {
    "ucmc-icu": ("UCMC", "mdl-fedavg10-mn", 15399),  # other two = mimic + nu
    "nu-icu": ("NU", "mdl-fedavg10-mc", 46030),  # other two = mimic + ucmc
    "mimic-icu": ("MIMIC", "mdl-fedavg10-cn", 24146),  # other two = ucmc + nu
}

# numerators (out of 100) of the fractions of each site's data that were used:
# 1..9 percent, then 10, 20, ..., 100 percent -> model suffix is the 3-digit value
frac_nums = list(range(1, 10)) + list(range(10, 101, 10))
fracs = [n / 100 for n in frac_nums]

# common lower bound for the (log) x-axes: smallest sample count plotted anywhere
x_lo = min(fracs) * min(size for *_, size in plot_dsets.values())

# shared, uniform ticks across all panels (plotly clips to each panel's range)
x_tickvals = [200, 500, 1000, 2000, 5000, 10000, 20000, 50000]
x_ticktext = ["200", "500", "1k", "2k", "5k", "10k", "20k", "50k"]

# number of federation rounds swept during training -> model suffix
fed_rounds = [1, 5, 10, 50, 100]

# dataset -> (display name, color); per-dataset palette, independent of the
# role-based palette used by the data-fraction plots above
round_dsets = {
    "ucmc-icu": ("UCMC", "#007396"),  # lake
    "nu-icu": ("NU", "#DE7C00"),  # terracotta
    "mimic-icu": ("MIMIC", "#275D38"),  # forest
}

# metric -> (results csv, axis/title label, output filename slug)
metrics = [
    ("tokenwise-roc-auc.csv", "ROC-AUC", "roc-auc"),
    ("tokenwise-pr-auc.csv", "PR-AUC", "pr-auc"),
]

for csv_name, metric_label, metric_slug in metrics:
    results = pd.read_csv(hm / csv_name)

    # -----------------------------------------------------------------------
    # performance vs. fraction of a site's training data, with fed baselines
    # -----------------------------------------------------------------------
    # aggregate over all tokens
    agg = results.set_index(["token", "models"]).groupby("models").mean()

    fig_width = 900
    s = fig_width / REF_WIDTH  # scale factor for all sized elements

    fig = make_subplots(
        rows=1,
        cols=len(plot_dsets),
        shared_yaxes=True,
        subplot_titles=[name for name, *_ in plot_dsets.values()],
        horizontal_spacing=0.03,
        # widths proportional to log-span up to each size -> uniform log x-scale
        column_widths=[
            math.log10(size) - math.log10(x_lo) for *_, size in plot_dsets.values()
        ],
    )
    fig.update_annotations(font_size=SUBTITLE_SIZE * s)  # subplot titles

    for col, (ds, (name, other_fed, size)) in enumerate(plot_dsets.items(), start=1):
        show = col == 1  # one shared legend

        # approximate training-set size at each fraction of this site's data
        sizes = [f * size for f in fracs]

        # site-trained sweep over increasing fractions of this site's data
        curve = [agg.loc[f"mdl-{ds}-{n:03d}", ds] for n in frac_nums]
        fig.add_trace(
            go.Scatter(
                x=sizes,
                y=curve,
                mode="lines+markers",
                name="site-trained",
                legendgroup="site-trained",
                showlegend=show,
                line=dict(color=COL_CURVE, width=LINE_WIDTH * s),
                marker=dict(size=MARKER_SIZE * s, color=COL_CURVE),
            ),
            row=1,
            col=col,
        )

        # horizontal federated / pooled baselines
        baselines = [
            ("fedavg (other two sites)", agg.loc[other_fed, ds], COL_OTHER),
            ("fedavg (all three sites)", agg.loc["mdl-fedavg10", ds], COL_FED),
            ("all data (pooled)", agg.loc["mdl-all", ds], COL_ALL),
        ]
        for label, val, color in baselines:
            fig.add_trace(
                go.Scatter(
                    x=[sizes[0], sizes[-1]],
                    y=[val, val],
                    mode="lines",
                    name=label,
                    legendgroup=label,
                    showlegend=show,
                    line=dict(color=color, width=LINE_WIDTH * s, dash="dash"),
                ),
                row=1,
                col=col,
            )
        fig.update_xaxes(
            title_text="training size",
            type="log",
            range=[math.log10(x_lo), math.log10(size)],
            tickvals=x_tickvals,
            ticktext=x_ticktext,
            row=1,
            col=col,
        )

    fig.update_yaxes(title_text=f"mean {metric_label}", row=1, col=1)
    fig.update_layout(
        template="plotly_white",
        font=dict(
            family="CMU Serif, Latin Modern Roman, serif",
            size=FONT_SIZE * s,
            color="black",
        ),
        title=dict(
            text=f"Mean {metric_label} vs. training set size",
            font=dict(size=TITLE_SIZE * s, color="black"),
        ),
        legend=dict(orientation="h", yanchor="top", y=-0.36, xanchor="center", x=0.5),
        margin=dict(l=60, r=30, t=120, b=210),
        width=fig_width,
        height=690,
    )
    fig.write_image(hm / f"data-fraction-sweep-{metric_slug}-aggregate.pdf")

    # -----------------------------------------------------------------------
    # average performance vs. number of federation rounds (all-site FedAvg)
    # -----------------------------------------------------------------------
    # one line per dataset, mean metric averaged over outcome tokens
    agg = results.set_index(["token", "models"]).groupby("models").mean()

    fig_width = 650
    s = fig_width / REF_WIDTH  # scale factor for all sized elements

    fig = go.Figure()
    for ds, (name, color) in round_dsets.items():
        curve = [agg.loc[f"mdl-fedavg{n}", ds] for n in fed_rounds]
        fig.add_trace(
            go.Scatter(
                x=fed_rounds,
                y=curve,
                mode="lines+markers",
                name=name,
                line=dict(color=color, width=LINE_WIDTH * s),
                marker=dict(size=MARKER_SIZE * s, color=color),
            )
        )

    fig.update_xaxes(
        title_text="federation rounds",
        type="log",
        tickvals=fed_rounds,
        ticktext=[str(n) for n in fed_rounds],
    )
    fig.update_yaxes(title_text=f"mean {metric_label}")
    fig.update_layout(
        template="plotly_white",
        font=dict(
            family="CMU Serif, Latin Modern Roman, serif",
            size=FONT_SIZE * s,
            color="black",
        ),
        title=dict(
            text=f"Mean {metric_label} vs. federation rounds",
            font=dict(size=TITLE_SIZE * s, color="black"),
        ),
        legend=dict(orientation="h", yanchor="top", y=-0.28, xanchor="center", x=0.5),
        margin=dict(l=60, r=30, t=60, b=150),
        width=fig_width,
        height=570,
    )
    fig.write_image(hm / f"fed-rounds-sweep-{metric_slug}.pdf")
