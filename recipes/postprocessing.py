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

from cotorra.util import bootstrap_ci

hm = (
    pathlib.Path("/gpfs/data" if os.uname().nodename.startswith("cri") else "/mnt")
    / "bbj-lab/users/burkh4rt"
)

dsets = ("mimic-icu", "ucmc-icu", "nu-icu", "all")

mdls = (
    [f"mdl-{ds}" for ds in dsets]
    + ["mdl-fedavg10"]
    + [f"mdl-{ds}-p" for ds in dsets]
    + ["mdl-fedavg10-p"]
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


def get_mdl_ds_res(ds, mdl, tt):
    df = pl.read_parquet(hm / "processed" / ds / mdl / "scores-rep-based-*.parquet")
    y_qual, y_true, y_score = (
        df.select(~pl.col(f"{tt}_past"), f"{tt}_future", f"{tt}_rep_score").to_numpy().T
    )
    # bootstrap_ci(
    #     y_true[y_qual.astype(bool)],
    #     np.nan_to_num(y_score)[y_qual.astype(bool)],
    #     n_samples=10,
    # )["roc_auc"].round(3)
    return skl_mets.roc_auc_score(
        y_true[y_qual.astype(bool)], np.nan_to_num(y_score)[y_qual.astype(bool)]
    )


for tt in grokked_outcome_tokens:
    print(f"--- {tt} ---")
    results = pd.DataFrame(columns=dsets, index=pd.Index(mdls, name="models"))
    for ds in dsets:
        for mdl in mdls:
            results.loc[mdl, ds] = get_mdl_ds_res(ds, mdl, tt)
    print(results)
