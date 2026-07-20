#!/usr/bin/env python3

"""
plot results
"""

import math
import pathlib

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

hm = pathlib.Path("~/Downloads").expanduser().resolve()
results = pd.read_csv(hm / "fed-results.csv")

agg = results.set_index(["token", "models"]).groupby("models").mean()

# ---------------------------------------------------------------------------
# performance vs. fraction of a site's training data, with federated baselines
# ---------------------------------------------------------------------------
# validated categorical palette (light mode)
COL_CURVE = "#007396"  # lake   -- site-trained sweep
COL_OTHER = "#DE7C00"  # terracotta -- fedavg on the other two sites
COL_FED = "#275D38"  # forest  -- fedavg on all three sites
COL_ALL = "#59315F"  # violet -- single model pooled over all data

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

# iterate over aggregate (None) then each individual token
tokens = [None] + list(results["token"].unique())

for token in tokens:
    if token is None:
        agg = results.set_index(["token", "models"]).groupby("models").mean()
    else:
        agg = (
            results[results["token"] == token]
            .set_index(["token", "models"])
            .groupby("models")
            .mean()
        )

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
                line=dict(color=COL_CURVE, width=2),
                marker=dict(size=8, color=COL_CURVE),
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
                    line=dict(color=color, width=2, dash="dash"),
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

    fig.update_yaxes(title_text="mean ROC AUC", row=1, col=1)
    fig.update_layout(
        template="plotly_white",
        font=dict(family="CMU Serif, Latin Modern Roman, serif", size=14),
        title="Performance vs. approx. size of site training data"
        + (f" — {token}" if token is not None else " (aggregate)"),
        legend=dict(
            orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5
        ),
        width=900,
        height=500,
    )
    safe_token = (
        "aggregate" if token is None else str(token).replace("/", "-").replace(" ", "_")
    )
    fig.write_image(hm / f"data-fraction-sweep-{safe_token}.pdf")
