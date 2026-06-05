#!/usr/bin/env python3

"""
collect results
"""

import pathlib

import numpy as np
import pandas as pd
import polars as pl
from omegaconf import OmegaConf

from cotorra.util import bootstrap_ci

hm = pathlib.Path("~/coreopsis").expanduser().resolve()

dsets = ["mimic-pre14", "mimic-post14", "ucmc-first"]
mdls = [f"mdl-{ds}" for ds in dsets] + ["mdl-fedavg10"]
tois = OmegaConf.load(hm / "src" / "coreopsis" / "config" / "scoring.yaml")[
    "tokens_of_interest"
]

for met in ["roc_auc", "pr_auc"]:  # "brier",
    print(f"=== {met} ===")
    for tt in tois:
        print(f"--- {tt} ---")
        results = pd.DataFrame(columns=dsets, index=pd.Index(mdls, name="models"))
        for ds in dsets:
            for mdl in mdls:
                df = pl.read_parquet(hm / "processed" / ds / mdl / "scores-*.parquet")
                y_qual, y_true, y_score = (
                    df.select(~pl.col(f"{tt}_past"), f"{tt}_future", f"{tt}_rep_score")
                    .to_numpy()
                    .T
                )
                results.loc[mdl, ds] = bootstrap_ci(
                    y_true[y_qual.astype(bool)],
                    np.nan_to_num(y_score)[y_qual.astype(bool)],
                    n_samples=1_000,
                )[met].round(3)  # add .mean() to drop CI's
        print(results)
