#!/usr/bin/env python3

"""
collect results
"""

import fnmatch
import importlib.resources as resources
import pathlib

import numpy as np
import pandas as pd
import polars as pl
from omegaconf import OmegaConf

from cotorra.util import bootstrap_ci

hm = pathlib.Path("~/coreopsis").expanduser().resolve()

dsets = [f"mimic-{y:02d}" for y in range(8, 21, 3)] + [
    f"ucmc-{y}" for y in range(18, 25)
]
mdls = [f"mdl-{ds}-p" for ds in dsets] + ["mdl-all-p", "mdl-fedavg10-p"]
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

for met in ["roc_auc"]:  # "brier", "pr_auc"
    print(f"=== {met} ===")
    for tt in grokked_outcome_tokens:
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
                results.loc[mdl, ds] = (
                    bootstrap_ci(
                        y_true[y_qual.astype(bool)],
                        np.nan_to_num(y_score)[y_qual.astype(bool)],
                        n_samples=1_00,
                    )[met]
                    .mean()
                    .round(3)
                )  # add  to drop CI's
        print(results)


# mimic_dsets = [f"mimic-{y:02d}" for y in range(8, 21, 3)]
# results = pd.DataFrame(
#     columns=grokked_outcome_tokens, index=pd.Index(mdls, name="models")
# )
# for mdl in mdls:
#     df = pl.read_parquet(
#         [hm / "processed" / ds / mdl / "scores-*.parquet" for ds in mimic_dsets]
#     )
#     for tt in grokked_outcome_tokens:
#         y_qual, y_true, y_score = (
#             df.filter(
#                 pl.col("race_category").str.to_lowercase().str.split(by=" ").list[0]
#                 == "black"
#             )
#             .select(~pl.col(f"{tt}_past"), f"{tt}_future", f"{tt}_rep_score")
#             .to_numpy()
#             .T
#         )
#         results.loc[mdl, tt] = (
#             bootstrap_ci(
#                 y_true[y_qual.astype(bool)],
#                 np.nan_to_num(y_score)[y_qual.astype(bool)],
#                 n_samples=1_000,
#             )[met]
#             .mean()
#             .round(3)
#         )
# print(results)


# ucmc_dsets = [f"ucmc-{y}" for y in range(18, 25)]
# results = pd.DataFrame(
#     columns=grokked_outcome_tokens, index=pd.Index(mdls, name="models")
# )
# for mdl in mdls:
#     df = pl.read_parquet(
#         [hm / "processed" / ds / mdl / "scores-*.parquet" for ds in ucmc_dsets]
#     )
#     for tt in grokked_outcome_tokens:
#         y_qual, y_true, y_score = (
#             df.filter(
#                 pl.col("race_category").str.to_lowercase().str.split(by=" ").list[0]
#                 == "white"
#             )
#             .select(~pl.col(f"{tt}_past"), f"{tt}_future", f"{tt}_rep_score")
#             .to_numpy()
#             .T
#         )
#         results.loc[mdl, tt] = (
#             bootstrap_ci(
#                 y_true[y_qual.astype(bool)],
#                 np.nan_to_num(y_score)[y_qual.astype(bool)],
#                 n_samples=1_000,
#             )[met]
#             .mean()
#             .round(3)
#         )
# print(results)
