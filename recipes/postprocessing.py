#!/usr/bin/env python3

"""
collect results
"""

import fnmatch
import importlib.resources as resources
import os
import pathlib
import typing
import warnings

import joblib as jl
import numpy as np
import pandas as pd
import polars as pl
from omegaconf import OmegaConf
from sklearn import metrics as skl_mets

from cotorra.util import bootstrap_aggregate_ci, bootstrap_aggregate_pval, pr_auc_score

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


def get_tokenwise_results(
    ds,
    mdl,
    inference: typing.Literal["rep-based", "generative"] = "rep-based",
    method: typing.Literal["mc", "scope", "reach", "rep"] = "rep",
):
    df = pl.read_parquet(hm / "processed" / ds / mdl / f"scores-{inference}-*.parquet")
    roc_auc, pr_auc = dict(), dict()
    for tt in grokked_outcome_tokens:
        y_qual, y_true, y_score = (
            df.select(~pl.col(f"{tt}_past"), f"{tt}_future", f"{tt}_{method}_score")
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


def get_all_tokenwise_results(
    dsets,
    mdls,
    inference: typing.Literal["rep-based", "generative"] = "rep-based",
    method: typing.Literal["mc", "scope", "reach", "rep"] = "rep",
):
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
                res, res_pr_auc = get_tokenwise_results(ds, mdl, inference, method)
                for tt in grokked_outcome_tokens:
                    results_roc_auc.loc[(tt, mdl), ds] = res[tt]
                    results_pr_auc.loc[(tt, mdl), ds] = res_pr_auc[tt]
            except (FileNotFoundError, pl.exceptions.ComputeError):
                pass  # no results for this dataset × model combination
    return results_roc_auc, results_pr_auc


def get_cis(
    ds,
    mdl,
    inference: typing.Literal["rep-based", "generative"] = "rep-based",
    method: typing.Literal["mc", "scope", "reach", "rep"] = "rep",
):
    df = pl.read_parquet(hm / "processed" / ds / mdl / f"scores-{inference}-*.parquet")
    y_trues, y_scores = [], []
    for tt in grokked_outcome_tokens:
        y_qual, y_true, y_score = (
            df.select(~pl.col(f"{tt}_past"), f"{tt}_future", f"{tt}_{method}_score")
            .to_numpy()
            .T
        )
        y_trues.append(y_true[y_qual.astype(bool)])
        y_scores.append(np.nan_to_num(y_score)[y_qual.astype(bool)])
    cis = bootstrap_aggregate_ci(
        y_trues, y_scores, n_samples=1_000, metrics=("avg_roc_auc", "avg_pr_auc")
    )
    return cis["avg_roc_auc"], cis["avg_pr_auc"]


def get_all_cis(
    dsets,
    mdls,
    inference: typing.Literal["rep-based", "generative"] = "rep-based",
    method: typing.Literal["mc", "scope", "reach", "rep"] = "rep",
):
    cis_roc_auc = pd.DataFrame(index=mdls, columns=dsets)
    cis_pr_auc = cis_roc_auc.copy()
    for mdl in mdls:
        for ds in dsets:
            try:
                cis_roc_auc.loc[mdl, ds], cis_pr_auc.loc[mdl, ds] = get_cis(
                    ds, mdl, inference, method
                )
            except (FileNotFoundError, pl.exceptions.ComputeError):
                pass
    return cis_roc_auc, cis_pr_auc


def get_diff_cis(
    ds,
    mdl0,
    mdl1,
    inference0: typing.Literal["rep-based", "generative"] = "rep-based",
    method0: typing.Literal["mc", "scope", "reach", "rep"] = "rep",
    inference1: typing.Literal["rep-based", "generative"] = "rep-based",
    method1: typing.Literal["mc", "scope", "reach", "rep"] = "rep",
    n_samples: int = 1_000,
    alpha: float = 0.05,
    rng: np.random.Generator = np.random.default_rng(seed=42),
):
    """
    Paired-bootstrap percentile interval for the difference in label-averaged
    performance between `mdl1` and the baseline `mdl0` (i.e. mdl1 − mdl0); both
    models are scored on the same subjects, so subjects are resampled as units
    to preserve the within-subject correlation between the two models.
    """
    df0 = pl.read_parquet(
        hm / "processed" / ds / mdl0 / f"scores-{inference0}-*.parquet"
    )
    df1 = pl.read_parquet(
        hm / "processed" / ds / mdl1 / f"scores-{inference1}-*.parquet"
    )
    y_trues, y_score0s, y_score1s = [], [], []
    for tt in grokked_outcome_tokens:
        y_qual0, y_true0, y_score0 = (
            df0.select(~pl.col(f"{tt}_past"), f"{tt}_future", f"{tt}_{method0}_score")
            .to_numpy()
            .T
        )
        y_qual1, y_true1, y_score1 = (
            df1.select(~pl.col(f"{tt}_past"), f"{tt}_future", f"{tt}_{method1}_score")
            .to_numpy()
            .T
        )
        assert np.array_equal(
            y_true0[y_qual0.astype(bool)], y_true1[y_qual1.astype(bool)]
        )
        y_trues.append(y_true0[y_qual0.astype(bool)])
        y_score0s.append(np.nan_to_num(y_score0)[y_qual0.astype(bool)])
        y_score1s.append(np.nan_to_num(y_score1)[y_qual1.astype(bool)])

    def get_diffs_i(rng_i: np.random.Generator) -> tuple[float, float]:
        warnings.filterwarnings("ignore")
        roc_diffs, pr_diffs = [], []
        for yt, s0, s1 in zip(y_trues, y_score0s, y_score1s):
            samp = rng_i.choice(len(yt), size=len(yt), replace=True)
            yti, s0i, s1i = yt[samp], s0[samp], s1[samp]
            roc_diffs.append(
                skl_mets.roc_auc_score(yti, s1i) - skl_mets.roc_auc_score(yti, s0i)
            )
            pr_diffs.append(pr_auc_score(yti, s1i) - pr_auc_score(yti, s0i))
        return np.mean(roc_diffs), np.mean(pr_diffs)

    with jl.Parallel(n_jobs=-1) as par:
        diffs = par(jl.delayed(get_diffs_i)(rng_i) for rng_i in rng.spawn(n_samples))
    return tuple(
        np.nanquantile([d[i] for d in diffs], q=[alpha / 2, 1 - (alpha / 2)])
        for i in (0, 1)
    )


def get_all_diff_cis(
    dsets,
    mdls,
    mdl_base,
    inference: typing.Literal["rep-based", "generative"] = "rep-based",
    method: typing.Literal["mc", "scope", "reach", "rep"] = "rep",
):
    """CIs for each model in `mdls` minus `mdl_base`, per dataset."""
    diffs_roc_auc = pd.DataFrame(index=mdls, columns=dsets)
    diffs_pr_auc = diffs_roc_auc.copy()
    for mdl in mdls:
        for ds in dsets:
            try:
                diffs_roc_auc.loc[mdl, ds], diffs_pr_auc.loc[mdl, ds] = get_diff_cis(
                    ds, mdl_base, mdl, inference, method, inference, method
                )
            except (FileNotFoundError, pl.exceptions.ComputeError):
                pass
    return diffs_roc_auc, diffs_pr_auc


def get_pvals(
    ds,
    mdl0,
    mdl1,
    alternative="two-sided",
    inference0: typing.Literal["rep-based", "generative"] = "rep-based",
    method0: typing.Literal["mc", "scope", "reach", "rep"] = "rep",
    inference1: typing.Literal["rep-based", "generative"] = "rep-based",
    method1: typing.Literal["mc", "scope", "reach", "rep"] = "rep",
):
    df0 = pl.read_parquet(
        hm / "processed" / ds / mdl0 / f"scores-{inference0}-*.parquet"
    )
    df1 = pl.read_parquet(
        hm / "processed" / ds / mdl1 / f"scores-{inference1}-*.parquet"
    )
    y_trues, y_score0s, y_score1s = [], [], []
    for tt in grokked_outcome_tokens:
        y_qual0, y_true0, y_score0 = (
            df0.select(~pl.col(f"{tt}_past"), f"{tt}_future", f"{tt}_{method0}_score")
            .to_numpy()
            .T
        )
        y_trues.append(y_true0[y_qual0.astype(bool)])
        y_score0s.append(np.nan_to_num(y_score0)[y_qual0.astype(bool)])
        y_qual1, y_true1, y_score1 = (
            df1.select(~pl.col(f"{tt}_past"), f"{tt}_future", f"{tt}_{method1}_score")
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
        paired=False,
        alternative=alternative,
    )
    return float(cis["avg_roc_auc"]), float(cis["avg_pr_auc"])


def re_fmt_ci(s, p=3):
    """Format a numpy CI (e.g. "[0.76768137, 0.79238952]") to p decimals."""
    lo, hi = min(s), max(s)
    return f"{(lo + hi) / 2:.{p}f} (±{(hi - lo) / 2:.{p}f})"


def mid(s, p=3):
    """Format a numpy CI (e.g. "[0.76768137, 0.79238952]") to p decimals."""
    lo, hi = min(s), max(s)
    return (lo + hi) / 2


def cat(df_roc, df_pr):
    return (
        pd.concat(
            (df_roc, df_pr),
            axis=1,
            keys=("ROC AUC", "PR AUC"),
            names=("metric", "test"),
        )
        .rename_axis("train")
        .map(re_fmt_ci)
    )


if __name__ == "__main__":
    tkwz_roc, tkwz_pr = get_all_tokenwise_results(
        dsets,
        [f"mdl-cxxx-{ds}-005" for ds in list(dsets) + ["all"]]
        + [f"mdl-c-{ds}" for ds in list(dsets) + ["all"]]
        + ["mdl-c-fedavg10", "mdl-c-fedavgm10", "mdl-c-fedadam10"],
    )
    tkwz_roc.to_csv(hm / "tkwz-roc.csv")
    tkwz_pr.to_csv(hm / "tkwz-pr.csv")

    xfer_roc, xfer_pr = get_all_cis(
        dsets,
        [f"mdl-c-{ds}" for ds in list(dsets)] + [f"mdl-cxxx-{ds}-005" for ds in dsets],
    )
    xfer_roc.map(mid)

    print(cat(xfer_roc, xfer_pr).to_latex(float_format="%.3f"))

    """
    transfer
    """
    xfer_roc, xfer_pr = get_all_cis(dsets, [f"mdl-c-{ds}" for ds in dsets])
    print(xfer_roc.map(re_fmt_ci).to_latex(float_format="%.3f"))
    print(xfer_pr.map(re_fmt_ci).to_latex(float_format="%.3f"))
    xfer_roc.to_csv(hm / "xfer-roc.csv")
    xfer_pr.to_csv(hm / "xfer-pr.csv")

    """
    federation strategy
    """
    mthd_roc, mthd_pr = get_all_cis(
        dsets,
        [f"mdl-c-{mthd}10" for mthd in ("fedavg", "fedavgm", "fedadam")]
        + ["mdl-c-all"],
    )
    print(mthd_roc.map(re_fmt_ci).to_latex(float_format="%.3f"))
    print(mthd_pr.map(re_fmt_ci).to_latex(float_format="%.3f"))
    mthd_roc.to_csv(hm / "mthd-roc.csv")
    mthd_pr.to_csv(hm / "mthd-pr.csv")

    """
    number of federation rounds
    """
    rnds_roc, rnds_pr = get_all_cis(dsets, [f"mdl-c-fedavg{i}" for i in (1, 5, 10, 50)])
    print(rnds_roc.map(re_fmt_ci).to_latex(float_format="%.3f"))
    print(rnds_pr.map(re_fmt_ci).to_latex(float_format="%.3f"))
    rnds_roc.to_csv(hm / "rnds-roc.csv")
    rnds_pr.to_csv(hm / "rnds-pr.csv")

    """
    fractional datasets / leave-one-dataset-out results
    """
    frac_roc, frac_pr = get_all_cis(
        dsets,
        [
            f"mdl-c-{ds}-{i:03d}"
            for ds in dsets
            for i in list(range(1, 11)) + list(range(15, 105, 5))
        ]
        + [f"mdl-c-fedavg10{sfx}" for sfx in ("", "-cn", "-mn", "-mc")]
        + ["mdl-c-all"],
    )
    frac_roc.to_csv(hm / "frac-roc.csv")
    frac_pr.to_csv(hm / "frac-pr.csv")
