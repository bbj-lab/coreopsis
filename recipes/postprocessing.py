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
from sklearn import metrics as skl_mets

from cotorra.util import bootstrap_aggregate_ci, bootstrap_aggregate_pval

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

dsets = ("ucmc-icu", "nu-icu", "mimic-icu")

mdls = list(
    {
        f"mdl-fedavg{n}{sfx}"
        for n in (1, 5, 10, 50, 100)
        for sfx in ("", "-mc", "-mn", "-cn")
    }
    | {
        f"mdl-{ds}-{i:03d}"
        for i in list(range(1, 11)) + list(range(20, 110, 10))
        for ds in dsets
    }
    | {
        f"mdl-{method}10{sfx}"
        for sfx in ("", "-mc", "-mn", "-cn")
        for method in ("fedavg", "fedavgm", "fedadam")
    }
    | {"mdl-all"}
)

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


def get_tokenwise_results(ds, mdl):
    df = pl.read_parquet(hm / "processed" / ds / mdl / "scores-rep-based-*.parquet")
    roc_auc, pr_auc = dict(), dict()
    for tt in grokked_outcome_tokens:
        y_qual, y_true, y_score = (
            df.select(~pl.col(f"{tt}_past"), f"{tt}_future", f"{tt}_rep_score")
            .to_numpy()
            .T
        )
        yt = y_true[y_qual.astype(bool)]
        ys = np.nan_to_num(y_score)[y_qual.astype(bool)]
        roc_auc[tt] = skl_mets.roc_auc_score(yt, ys)
        precs, recs, _ = skl_mets.precision_recall_curve(
            yt, np.round(ys, decimals=4), drop_intermediate=True
        )
        pr_auc[tt] = skl_mets.auc(recs, precs)
    return roc_auc, pr_auc


def get_all_tokenwise_results(dsets, mdls):
    results_roc_auc = pd.DataFrame(
        index=pd.MultiIndex.from_product(
            (grokked_outcome_tokens, mdls), names=("token", "models")
        ),
        columns=dsets,
    )
    results_pr_auc = results_roc_auc.copy()
    for mdl in mdls:
        for ds in dsets:
            try:
                res, res_pr_auc = get_tokenwise_results(ds, mdl)
                for tt in grokked_outcome_tokens:
                    results_roc_auc.loc[(tt, mdl), ds] = res[tt]
                    results_pr_auc.loc[(tt, mdl), ds] = res_pr_auc[tt]
            except (FileNotFoundError, pl.exceptions.ComputeError):
                pass  # no results for this dataset × model combination
    return results_roc_auc, results_pr_auc


def get_cis(ds, mdl):
    df = pl.read_parquet(hm / "processed" / ds / mdl / "scores-rep-based-*.parquet")
    y_trues, y_scores = [], []
    for tt in grokked_outcome_tokens:
        y_qual, y_true, y_score = (
            df.select(~pl.col(f"{tt}_past"), f"{tt}_future", f"{tt}_rep_score")
            .to_numpy()
            .T
        )
        y_trues.append(y_true[y_qual.astype(bool)])
        y_scores.append(np.nan_to_num(y_score)[y_qual.astype(bool)])
    cis = bootstrap_aggregate_ci(
        y_trues, y_scores, n_samples=1_000, metrics=("avg_roc_auc", "avg_pr_auc")
    )
    return cis["avg_roc_auc"], cis["avg_pr_auc"]


def get_all_cis(dsets, mdls):
    cis_roc_auc = pd.DataFrame(index=mdls, columns=dsets)
    cis_pr_auc = cis_roc_auc.copy()
    for mdl in mdls:
        for ds in dsets:
            try:
                cis_roc_auc.loc[mdl, ds], cis_pr_auc.loc[mdl, ds] = get_cis(ds, mdl)
            except (FileNotFoundError, pl.exceptions.ComputeError):
                pass
    return cis_roc_auc, cis_pr_auc


def get_pvals(ds, mdl0, mdl1):
    df0 = pl.read_parquet(hm / "processed" / ds / mdl0 / "scores-rep-based-*.parquet")
    df1 = pl.read_parquet(hm / "processed" / ds / mdl1 / "scores-rep-based-*.parquet")
    y_trues, y_score0s, y_score1s = [], [], []
    for tt in grokked_outcome_tokens:
        y_qual0, y_true0, y_score0 = (
            df0.select(~pl.col(f"{tt}_past"), f"{tt}_future", f"{tt}_rep_score")
            .to_numpy()
            .T
        )
        y_trues.append(y_true0[y_qual0.astype(bool)])
        y_score0s.append(np.nan_to_num(y_score0)[y_qual0.astype(bool)])
        y_qual1, y_true1, y_score1 = (
            df1.select(~pl.col(f"{tt}_past"), f"{tt}_future", f"{tt}_rep_score")
            .to_numpy()
            .T
        )
        assert np.array_equal(
            y_true0[y_qual0.astype(bool)], y_true1[y_qual1.astype(bool)]
        )
        y_score1s.append(np.nan_to_num(y_score1)[y_qual1.astype(bool)])
    cis = bootstrap_aggregate_pval(
        y_trues,
        y_score0s,
        y_score1s,
        n_samples=1_000,
        metrics=("avg_roc_auc", "avg_pr_auc"),
        paired=True,
        alternative="two-sided",
    )
    return float(cis["avg_roc_auc"]), float(cis["avg_pr_auc"])


if __name__ == "__main__":
    tokenwise_roc_auc, tokenwise_pr_auc = get_all_tokenwise_results(dsets, mdls)
    tokenwise_roc_auc.to_csv(hm / "tokenwise-roc-auc.csv")
    tokenwise_pr_auc.to_csv(hm / "tokenwise-pr-auc.csv")

    aggregate_roc_cis, aggregate_pr_cis = get_all_cis(dsets, mdls)
    aggregate_roc_cis.to_csv(hm / "aggregate-roc-cis.csv")
    aggregate_pr_cis.to_csv(hm / "aggregate-pr-cis.csv")
