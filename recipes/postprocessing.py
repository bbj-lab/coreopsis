#!/usr/bin/env python3

"""
collect results
"""

import fnmatch
import importlib.resources as resources
import os
import pathlib

import numpy as np
import pandas as pd
import polars as pl
from omegaconf import OmegaConf
from sklearn import linear_model as skl_lm
from sklearn import metrics as skl_mets
from sklearn import model_selection as skl_mdl_sel

pd.options.display.float_format = "{:,.3f}".format
pd.options.display.max_columns = None
pd.options.display.max_rows = 100
pd.options.display.width = None
pd.options.display.expand_frame_repr = False
pd.options.display.show_dimensions = True

hm = (
    pathlib.Path("/gpfs/data" if os.uname().nodename.startswith("cri") else "/mnt")
    / "bbj-lab/users/burkh4rt"
)

dsets = ("mimic-icu", "ucmc-icu", "nu-icu")

mdls = [f"mdl-{ds}-{i}" for ds in dsets for i in range(1, 11)] + [
    "mdl-fedavg10",
    "mdl-fedavg10-mc",
    "mdl-fedavg10-mn",
    "mdl-fedavg10-cn",
    "mdl-all",
]
grokked_outcome_tokens = [
    x
    for x in OmegaConf.load(
        hm / "processed" / dsets[0] / "tokenizer.yaml"
    ).lookup.keys()
    if any(
        fnmatch.fnmatch(x, p)
        for p in OmegaConf.load(resources.files("coreopsis.config") / "scoring.yaml")[
            "tokens_of_interest"
        ]
    )
]


def get_stacked_results(ds, mdls_to_stack):
    df = pl.concat(
        [
            pl.read_parquet(
                hm / "processed" / ds / mdl / "scores-rep-based-*.parquet"
            ).with_columns(mdl=pl.lit(mdl))
            for mdl in mdls_to_stack
        ],
        how="diagonal",
    )
    res = dict()
    for tt in grokked_outcome_tokens:
        preds = np.hstack(
            [
                df.filter(mdl=mdl).select(f"{tt}_rep_score").to_numpy()
                for mdl in mdls_to_stack
            ]
        )
        y_qual, y_true = (
            df.filter(mdl=mdls_to_stack[0])
            .select(~pl.col(f"{tt}_past"), f"{tt}_future")
            .to_numpy()
            .T
        )
        stacked = np.nan * np.ones_like(y_true)
        stacked[y_qual] = skl_mdl_sel.cross_val_predict(
            skl_lm.LogisticRegressionCV(
                n_jobs=-1,
                scoring="roc_auc",
                max_iter=10_000,
                use_legacy_attributes=False,
                l1_ratios=(0,),
            ),
            X=preds[y_qual],
            y=y_true[y_qual],
            method="predict_proba",
        )[:, 1]
        res[tt] = skl_mets.roc_auc_score(
            y_true[y_qual.astype(bool)], np.nan_to_num(stacked)[y_qual.astype(bool)]
        )
    return res


def get_results(ds, mdl, metric: str = "roc_auc_score"):
    df = pl.read_parquet(hm / "processed" / ds / mdl / "scores-rep-based-*.parquet")
    res = dict()
    for tt in grokked_outcome_tokens:
        y_qual, y_true, y_score = (
            df.select(~pl.col(f"{tt}_past"), f"{tt}_future", f"{tt}_rep_score")
            .to_numpy()
            .T
        )
        yt = y_true[y_qual.astype(bool)]
        ys = np.nan_to_num(y_score)[y_qual.astype(bool)]
        if metric == "pr_auc_score":
            precs, recs, _ = skl_mets.precision_recall_curve(
                yt, np.round(ys, decimals=4), drop_intermediate=True
            )
            res[tt] = skl_mets.auc(recs, precs)
        else:
            res[tt] = getattr(skl_mets, metric)(yt, ys)
    return res


results = pd.DataFrame(
    columns=dsets,
    index=pd.MultiIndex.from_product(
        (grokked_outcome_tokens, mdls), names=("token", "models")
    ),
)

for ds in dsets:
    for mdl in mdls + [
        f"mdl-{ds}-{i:03d}" for i in list(range(1, 11)) + list(range(20, 110, 10))
    ]:
        res = get_results(ds, mdl)
        for tt in grokked_outcome_tokens:
            results.loc[(tt, mdl), ds] = res[tt]

results.to_csv(hm / "fed-results.csv")


mdls = ["mdl-mimic-icu-100", "mdl-ucmc-icu-100", "mdl-nu-icu-100", "mdl-all"]
xfer = pd.DataFrame(
    columns=dsets,
    index=pd.MultiIndex.from_product(
        (grokked_outcome_tokens, mdls), names=("token", "models")
    ),
)
for ds in dsets:
    for mdl in mdls:
        res = get_results(ds, mdl)
        for tt in grokked_outcome_tokens:
            xfer.loc[(tt, mdl), ds] = res[tt]

agg_xfer = xfer.groupby("models").mean().sort_values("models")
xfer.to_csv(hm / "xfer-results.csv")

for ds in dsets:
    for mdl in mdls:
        res = get_results(ds, mdl, "pr_auc_score")
        for tt in grokked_outcome_tokens:
            xfer.loc[(tt, mdl), ds] = res[tt]

agg_xfer = xfer.groupby("models").mean().sort_values("models")
