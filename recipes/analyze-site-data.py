#!/usr/bin/env python3

"""
quantify how far apart the sites are: cross-site divergence of the quantized
token distributions, and cross-site next-token loss of the single-site models
"""

import os
import pathlib
from itertools import combinations

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import polars as pl
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from plotly.subplots import make_subplots
from transformers import AutoModelForCausalLM

pl.Config(set_fmt_float="mixed", float_precision=3, tbl_cols=-1)

hm = (
    pathlib.Path("/gpfs/data" if os.uname().nodename.startswith("cri") else "/mnt")
    / "bbj-lab/users/burkh4rt"
)

colors = {
    # primary
    "maroon": "#800000",
    "light_greystone": "#D9D9D9",
    "greystone": "#A6A6A6",
    "dark_greystone": "#737373",
    "white": "#FFFFFF",
    "black": "#000000",
    # secondary - base shades
    "goldenrod": "#EAAA00",
    "terracotta": "#DE7C00",
    "ivy": "#789D4A",
    "forest": "#275D38",
    "lake": "#007396",
    "violet": "#59315F",
    "brick": "#A4343A",
    # secondary - light shades
    "goldenrod_light": "#F3D03E",
    "terracotta_light": "#ECA154",
    "ivy_light": "#A9C47F",
    "forest_light": "#9CAF88",
    "lake_light": "#3EB1C8",
    "violet_light": "#86647A",
    "brick_light": "#B46A55",
    # secondary - dark shades
    "goldenrod_dark": "#CC8A00",
    "terracotta_dark": "#A9431E",
    "ivy_dark": "#13301C",
    "forest_dark": "#284734",
    "lake_dark": "#002A3A",
    "violet_dark": "#41273B",
    "brick_dark": "#643335",
}

# dataset -> (display name, color); the same per-site palette plotting.py gives
# the round sweep, so a site keeps its color from figure to figure. this dict
# also fixes the order the sites appear in everywhere below
sites = {
    "ucmc-icu": ("UCMC", "#800000"),
    "nu-icu": ("NU", "#4E2A84"),
    "rush-icu": ("RUSH", "#006937"),
    "mimic-icu": ("MIMIC", "#0069b4"),
}
n_sites = len(sites)


def tint(hex_color, frac):
    """Blend '#RRGGBB' `frac` of the way toward white."""
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i : i + 2], 16) for i in (0, 2, 4))
    m = lambda c: round(c + (255 - c) * frac)  # noqa: E731
    return f"#{m(r):02X}{m(g):02X}{m(b):02X}"


"""
per-site token statistics
"""


def load_vocab(ds):
    """token id -> token string. the tokenizer is learned once and applied to
    every site, but each processed directory carries its own copy."""
    lookup = OmegaConf.load(hm / "processed" / ds / "tokenizer.yaml").lookup
    return {token_id: token for token, token_id in lookup.items()}


def token_stats(ds):
    """how often each token occurs in a site's timelines, both as a share of
    all that site's tokens (frequency) and as a share of its timelines
    (prevalence)"""
    lf = pl.scan_parquet(hm / "processed" / ds / "tokens_times.parquet").with_columns(
        pl.col("tokens").list.eval(
            pl.element().replace_strict(load_vocab(ds), return_dtype=pl.String)
        )
    )
    n_timelines = lf.select(pl.len()).collect().item()
    return (
        lf.select("subject_id", "tokens")
        .explode("tokens", empty_as_null=True)
        .rename({"tokens": "token"})
        .group_by("token")
        .agg(
            pl.len().alias("n_occurrences"),
            pl.col("subject_id").n_unique().alias("n_timelines_with_token"),
        )
        .with_columns(
            (pl.col("n_timelines_with_token") / n_timelines).alias("prevalence"),
            (pl.col("n_occurrences") / pl.col("n_occurrences").sum()).alias(
                "frequency"
            ),
        )
        .sort("token")
        .collect(engine="streaming")
    )


token_stats_by_site = pl.concat(
    token_stats(ds).with_columns(pl.lit(ds).alias("site")) for ds in sites
)
token_stats_by_site.write_csv(hm / "site-token-stats.csv")
print(token_stats_by_site)

"""
cross-site divergence of the quantized measurements
"""

# tokens ending in _Q0.._Q9 are the quantile-binned measurements; each
# `TYPE//name` group is a distribution over its own ordered bins, so the same
# group at two sites is directly comparable
qtokens = token_stats_by_site.filter(
    pl.col("token").str.contains(r"_Q\d$")
).with_columns(pl.col("token").str.extract(r"^(.+)_Q\d$", 1).alias("group"))

n_per_col = 5  # panels per column: the n most and n least divergent groups


def wasserstein1(p, q):
    """1-Wasserstein distance between two distributions over ordered bins of
    unit spacing. unlike a KL divergence this charges for *how far* mass moved,
    which is what we want when the bins are quantiles of one measurement."""
    return float(np.abs(np.cumsum(p) - np.cumsum(q)).sum())


dists = {}  # group -> (bin labels, {dataset: distribution over those bins})
divergence = {}  # group -> mean pairwise W1 across the sites

for group in sorted(qtokens["group"].unique().to_list()):
    wide = (
        qtokens.filter(pl.col("group") == group)
        .sort("token")
        .pivot(values="frequency", index="token", on="site")
    )
    if not all(ds in wide.columns for ds in sites):
        continue  # group is absent at a site, so there is nothing to compare

    # renormalize each site's frequencies within the group: the comparison is
    # between the *shapes* of the binned distributions, not between how often
    # the underlying measurement gets recorded at each site
    p = {}
    for ds in sites:
        col = wide[ds].fill_null(0.0).to_numpy().astype(float)
        if col.sum() > 0:
            p[ds] = col / col.sum()
    if len(p) < n_sites:
        continue

    dists[group] = ([t.rsplit("_", 1)[-1] for t in wide["token"]], p)
    divergence[group] = float(
        np.mean([wasserstein1(p[a], p[b]) for a, b in combinations(p, 2)])
    )

ranked = sorted(divergence, key=divergence.get, reverse=True)
# most-divergent first in the left column, most-similar first in the right
columns = (
    ("Highest Avg. Cross-Site Wasserstein-1", ranked[:n_per_col]),
    ("Lowest Avg. Cross-Site Wasserstein-1", ranked[::-1][:n_per_col]),
)

fig_width = 1350  # 50% wider than the 900px reference (see plotting.py)
s = 650 / 900  # scale factor for all sized elements
h_space = 0.09  # gap between the two columns of panels, as a paper fraction

# row-major panel order, matching how make_subplots fills `subplot_titles`
panels = [
    (row, col, group)
    for col, (_, groups) in enumerate(columns, start=1)
    for row, group in enumerate(groups, start=1)
]
titles = [""] * (2 * n_per_col)
for row, col, group in panels:
    titles[2 * (row - 1) + col - 1] = f"{group} (W₁ = {divergence[group]:.3f})"

fig = make_subplots(
    rows=n_per_col,
    cols=2,
    subplot_titles=titles,
    horizontal_spacing=h_space,
    vertical_spacing=0.055,
)
fig.update_annotations(font_size=26 * s)  # subplot (panel) titles

for row, col, group in panels:
    bins, p = dists[group]
    for ds, (name, color) in sites.items():
        fig.add_trace(
            go.Bar(
                x=bins,
                y=p[ds],
                name=name,
                legendgroup=name,
                showlegend=(row == 1 and col == 1),  # one shared legend
                marker=dict(color=color, line=dict(width=0)),
            ),
            row=row,
            col=col,
        )

# label the middle row of each column only, so the one title reads as covering
# the column (and the bottom row, for the shared bin axis)
for col in (1, 2):
    fig.update_yaxes(title_text="share of group", row=(n_per_col + 1) // 2, col=col)
    fig.update_xaxes(title_text="quantile bin", row=n_per_col, col=col)
fig.update_xaxes(tickangle=0)

# column headers, above the top row's panel titles
for x, (header, _) in zip((0.25 - h_space / 4, 0.75 + h_space / 4), columns):
    fig.add_annotation(
        text=header,
        xref="paper",
        yref="paper",
        x=x,
        y=1.028,
        xanchor="center",
        yanchor="bottom",
        showarrow=False,
        font=dict(size=34 * s, color="black"),
    )

fig.update_layout(
    template="plotly_white",
    barmode="group",
    bargap=0.25,
    bargroupgap=0.03,  # keeps a sliver of surface between neighboring bars
    font=dict(
        family="Gotham Book, Gotham, Helvetica, sans-serif", size=30 * s, color="black"
    ),
    title=None,
    legend=dict(orientation="v", yanchor="middle", y=0.5, xanchor="left", x=1.02),
    margin=dict(l=60, r=30, t=230, b=100),
    width=fig_width,
    height=2000,
)
fig.write_image(hm / "W1-distribution-analysis.pdf")

"""
cross-site next-token loss of the single-site models
"""

# every site supplies eval timelines, but a site only supplies a row of the
# matrix if a single-site model was trained on it; rush-icu has none, so it
# appears as an eval column only
eval_only = ("rush-icu",)
model_sites = {ds: v for ds, v in sites.items() if ds not in eval_only}
n_models = len(model_sites)
# columns keep the usual site order, except that the eval-only sites are moved
# to the right of the ones that also supply a row
eval_sites = model_sites | {ds: v for ds, v in sites.items() if ds in eval_only}

device = "cuda" if torch.cuda.is_available() else "cpu"
n_sampled = 5000  # timelines sampled per site
max_len = 4096  # context window; longer timelines are truncated


def mean_nll(model, timelines):
    """mean next-token negative log likelihood over a list of token-id lists"""
    total, n_tokens = 0.0, 0
    with torch.no_grad():
        for tokens in timelines:
            ids = torch.tensor(
                tokens[:max_len], dtype=torch.long, device=device
            ).unsqueeze(0)
            if ids.shape[1] < 2:
                continue
            nll = F.cross_entropy(
                model(ids).logits[0, :-1], ids[0, 1:], reduction="none"
            )
            total += nll.sum().item()
            n_tokens += nll.numel()
    return total / n_tokens


# sampled once per site and reused across models, so every row of the matrix
# below is scored on identical timelines
samples = {
    ds: pl.read_parquet(
        hm / "processed" / ds / "tokens_times.parquet", columns=["tokens"]
    )
    .sample(n=n_sampled, seed=0)["tokens"]
    .to_list()
    for ds in sites
}

nll = {}
for model_ds in model_sites:
    model = (
        AutoModelForCausalLM.from_pretrained(
            hm / "output" / f"c-{model_ds}" / "mdl-cotorra"
        )
        .to(device)
        .eval()
    )
    for eval_ds in eval_sites:
        nll[model_ds, eval_ds] = mean_nll(model, samples[eval_ds])
    del model
    if device == "cuda":
        torch.cuda.empty_cache()

nll_matrix = pl.DataFrame(
    [
        {"model_site": model_ds}
        | {eval_ds: nll[model_ds, eval_ds] for eval_ds in eval_sites}
        for model_ds in model_sites
    ]
)
nll_matrix.write_csv(hm / "cross-site-nll.csv")
print(nll_matrix)

s = 650 / 900
cell = 142  # px per cell, so the cells stay square as sites are added
margin = dict(l=150, r=40, t=170, b=40)

m = np.array(
    [[nll[model_ds, eval_ds] for eval_ds in eval_sites] for model_ds in model_sites]
)

# the matrix itself is the only thing that gets a fill; the trailing mean row
# and column are left unfilled (nan) so they read as marginals rather than as
# more cells on the same scale
z = np.full((n_models + 1, n_sites + 1), np.nan)
z[:n_models, :n_sites] = m
shown = z.copy()
shown[:n_models, n_sites] = m.mean(axis=1)  # per model, over eval sites
shown[n_models, :n_sites] = m.mean(axis=0)  # per eval site, over models
shown[n_models, n_sites] = m.mean()

x_labels = [name for name, _ in eval_sites.values()] + ["mean"]
y_labels = [name for name, _ in model_sites.values()] + ["mean"]
lo, hi = m.min(), m.max()
text_flips = 0.5  # normalized value above which cell text goes white

fig = go.Figure(
    go.Heatmap(
        z=z,
        x=x_labels,
        y=y_labels,
        # single-hue sequential ramp, light -> dark, drawn from the lake family
        colorscale=px.colors.sequential.Viridis,
        showscale=False,  # every cell is labeled, so a colorbar adds nothing
        xgap=2,
        ygap=2,
        hoverinfo="skip",
    )
)

for i in range(n_models + 1):
    for j in range(n_sites + 1):
        v = shown[i, j]
        in_matrix = i < n_models and j < n_sites
        # the matrix is no longer square, so "a model on its own site" is a
        # match between the row's and the column's dataset, not i == j
        own_site = in_matrix and list(model_sites)[i] == list(eval_sites)[j]
        # bold the own-site cells and the marginals
        fig.add_annotation(
            x=x_labels[j],
            y=y_labels[i],
            text=f"{v:.3f}" if in_matrix and not own_site else f"<b>{v:.3f}</b>",
            showarrow=False,
            font=dict(
                size=24 * s,
                color=colors["white"]
                if in_matrix and (v - lo) / (hi - lo) > text_flips
                else colors["black"],
            ),
        )

# a hairline greystone rule on every cell boundary, so each number sits in its
# own box. these have to be shapes: a categorical axis draws its gridlines
# through the category centers, not between them. the marginals still get no
# rule of their own setting them off -- they are the only unfilled cells in the
# grid, which already reads as a break
x_edges = [i - 0.5 for i in range(n_sites + 2)]  # boundaries, outer included
y_edges = [i - 0.5 for i in range(n_models + 2)]
rule = dict(line=dict(color=colors["greystone"], width=1.5), layer="above")
for e in x_edges:
    fig.add_shape(type="line", x0=e, x1=e, y0=y_edges[0], y1=y_edges[-1], **rule)
for e in y_edges:
    fig.add_shape(type="line", x0=x_edges[0], x1=x_edges[-1], y0=e, y1=e, **rule)

fig.update_xaxes(
    side="top", title_text="eval site", showgrid=False, zeroline=False, ticks=""
)
fig.update_yaxes(
    autorange="reversed",  # first model site at the top
    title_text="model site",
    showgrid=False,
    zeroline=False,
    ticks="",
)
fig.update_layout(
    template="plotly_white",
    font=dict(
        family="Gotham Book, Gotham, Helvetica, sans-serif", size=30 * s, color="black"
    ),
    title=None,
    # the plotting area is sized off the cell count rather than fixed, since
    # there is one fewer model site than eval site
    margin=margin,
    width=cell * (n_sites + 1) + margin["l"] + margin["r"],
    height=cell * (n_models + 1) + margin["t"] + margin["b"],
)
fig.write_image(hm / "cross-site-nll.pdf")
