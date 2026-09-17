#!/usr/bin/env python3

"""
baseline model
"""

import fnmatch
import importlib.resources as resources
import os
import pathlib

import lightgbm as lgb
import numpy as np
import pandas as pd
import polars as pl
import sklearn as skl
from omegaconf import OmegaConf
from sklearn import metrics as skl_mets

from cotorra.util import bootstrap_aggregate_ci

pd.options.display.float_format = "{:,.3f}".format
pd.options.display.max_columns = None
pd.options.display.max_rows = 100
pd.options.display.width = None
pd.options.display.expand_frame_repr = False
pd.options.display.show_dimensions = True
pd.set_option("performance_warnings", False)

hm = (
    pathlib.Path("/gpfs/data" if os.uname().nodename.startswith("cri") else "/mnt")
    / "bbj-lab/users/burkh4rt"
)

dsets = os.getenv("dsets_csv").strip(",").split(",")
vocab = OmegaConf.load(hm / "processed" / dsets[0] / "tokenizer.yaml").lookup

grokked_outcome_tokens = [
    x
    for x in vocab.keys()
    if any(
        fnmatch.fnmatch(x, p)
        for p in OmegaConf.load(resources.files("coreopsis.config") / "scoring.yaml")[
            "tokens_of_interest"
        ]
    )
]


def re_fmt_ci(s, p=3):
    """Format a numpy CI (e.g. "[0.76768137, 0.79238952]") to p decimals."""
    lo, hi = min(s), max(s)
    return f"{(lo + hi) / 2:.{p}f} (±{(hi - lo) / 2:.{p}f})"


bl_roc_auc = pd.DataFrame(
    index=pd.MultiIndex.from_product(
        (grokked_outcome_tokens, dsets + ["all"]), names=("token", "train")
    ),
    columns=dsets,
)
bl_pr_auc = bl_roc_auc.copy()

bl_roc_agg = pd.DataFrame(index=dsets + ["all"], columns=dsets)
bl_pr_agg = bl_roc_agg.copy()

df_train, df_tuning, df_held_out, mdls = dict(), dict(), dict(), dict()

for lgbm in (True, False):
    for counts in (True, False):
        model = "LGBM" if lgbm else "LR"
        method = "C" if counts else "1H"
        print(f"{model}-{method}")
        for ds in dsets:
            lf_train, lf_tuning, lf_held_out = (
                pl.scan_parquet(
                    hm / "processed" / ds / f"{split}_for_inference.parquet"
                ).with_columns(
                    pl.col("tokens_past").list.len().alias("n_tokens_first_24h"),
                    *[
                        pl.col("tokens_past").list.count_matches(v).alias(f"count_{v}")
                        if counts
                        else pl.col("tokens_past").list.contains(v).alias(f"has_{v}")
                        for v in vocab.values()
                    ],
                )
                for split in ("train", "tuning", "held_out")
            )
            for tt in grokked_outcome_tokens:
                df_train[(tt, ds)], df_tuning[(tt, ds)], df_held_out[(tt, ds)] = (
                    lf.filter(~pl.col(f"{tt}_past"))
                    .select(
                        *[
                            f"count_{v}" if counts else f"has_{v}"
                            for v in vocab.values()
                        ],
                        f"{tt}_future",
                    )
                    .collect()
                    .to_pandas()
                    for lf in (lf_train, lf_tuning, lf_held_out)
                )
                mdls[(tt, ds)] = (
                    lgb.LGBMClassifier(n_jobs=-1)
                    if lgbm
                    else skl.linear_model.LogisticRegression(max_iter=10_000)
                )
                mdls[(tt, ds)].fit(
                    X=df_train[(tt, ds)].drop(columns=f"{tt}_future"),
                    y=df_train[(tt, ds)][f"{tt}_future"].astype(int),
                )

        for tt in grokked_outcome_tokens:
            df_train_all = pd.concat(df_train[(tt, ds)] for ds in dsets)
            df_tuning_all = pd.concat(df_tuning[(tt, ds)] for ds in dsets)
            mdls[(tt, "all")] = (
                lgb.LGBMClassifier(n_jobs=-1)
                if lgbm
                else skl.linear_model.LogisticRegression(max_iter=10_000)
            )
            mdls[(tt, "all")].fit(
                X=df_train_all.drop(columns=f"{tt}_future"),
                y=df_train_all[f"{tt}_future"].astype(int),
            )

        for ds_train in dsets + ["all"]:
            for ds_test in dsets:
                y_trues, y_scores = [], []
                for tt in grokked_outcome_tokens:
                    yt = df_held_out[(tt, ds_test)][f"{tt}_future"].astype(int)
                    ys = mdls[(tt, ds_train)].predict_proba(
                        df_held_out[(tt, ds_test)].drop(columns=f"{tt}_future")
                    )[:, 1]
                    bl_roc_auc.loc[(tt, ds_train), ds_test] = skl_mets.roc_auc_score(
                        yt, ys
                    )
                    precs, recs, _ = skl_mets.precision_recall_curve(
                        yt, np.round(ys, decimals=4), drop_intermediate=True
                    )
                    bl_pr_auc.loc[(tt, ds_train), ds_test] = skl_mets.auc(recs, precs)
                    y_trues.append(yt)
                    y_scores.append(ys)
                cis = bootstrap_aggregate_ci(
                    y_trues,
                    y_scores,
                    n_samples=1_000,
                    metrics=("avg_roc_auc", "avg_pr_auc"),
                )
                bl_roc_agg.loc[ds_train, ds_test] = cis["avg_roc_auc"]
                bl_pr_agg.loc[ds_train, ds_test] = cis["avg_pr_auc"]

        print(
            pd.concat(
                (bl_roc_agg, bl_pr_agg),
                axis=1,
                keys=("ROC AUC", "PR AUC"),
                names=("metric", "test"),
            )
            .rename_axis("train")
            .map(re_fmt_ci)
            .to_latex(float_format="%.3f")
        )

        bl_roc_auc.to_csv(hm / f"bl-roc-auc-{model}-{method}.csv")
        bl_pr_auc.to_csv(hm / f"bl-pr-auc-{model}-{method}.csv")

        bl_roc_agg.to_csv(hm / f"bl-roc-agg-{model}-{method}.csv")
        bl_pr_agg.to_csv(hm / f"bl-pr-agg-{model}-{method}.csv")
