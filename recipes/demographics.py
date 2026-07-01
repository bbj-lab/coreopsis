#!/usr/bin/env python3

"""
gather stats on training sets
"""

import importlib.resources as resources
import os
import pathlib

import polars as pl
from omegaconf import OmegaConf

pl.Config(set_fmt_float="mixed", float_precision=3, tbl_rows=-1, tbl_cols=-1)

hm = (
    pathlib.Path("/gpfs/data" if os.uname().nodename.startswith("cri") else "/mnt")
    / "bbj-lab/users/burkh4rt"
)

dsets = ("mimic-icu", "ucmc-icu", "nu-icu", "all")

pt_cols = OmegaConf.load(resources.files("coreopsis.config") / "collation.yaml")[
    "pass_through_columns"
]
df = pl.read_parquet(
    list((hm / "processed" / dsets[0]).glob("*_for_inference.parquet"))
)
labels = [
    c[:-5] for c, t in df.schema.items() if c.endswith("_past") and t == pl.Boolean
]


df_outcomes = pl.concat(
    [
        pl.read_parquet(list((hm / "processed" / ds).glob("*_for_inference.parquet")))
        .with_columns(
            [
                pl.any_horizontal(pl.col(f"{c}_past", f"{c}_future")).alias(c)
                for c in labels
            ]
        )
        .select(
            [pl.lit(ds).alias("dataset")]
            + [pl.col(c).mean().alias(c.split("//")[-1]) for c in labels]
        )
        for ds in dsets
    ]
)

print(
    df_outcomes.transpose(
        include_header=True, header_name="outcome", column_names="dataset"
    )
    # .to_pandas()
    # .to_latex(index=False, float_format="%.3f")
)

df_demog = pl.concat(
    [
        pl.read_parquet(
            list((hm / "processed" / ds).glob("*_for_inference.parquet"))
        ).select(
            pl.lit(ds).alias("dataset"),
            pl.len().alias("count"),
            pl.col("age_at_admission").mean().alias("age_avg"),
            pl.col("age_at_admission").std().alias("age_std"),
            (pl.col("sex_category").str.to_lowercase() == "female")
            .mean()
            .alias("freq_female"),
            (pl.col("ethnicity_category").str.to_lowercase() == "hispanic")
            .mean()
            .alias("freq_hispanic"),
            *[
                (pl.col("language_category").str.to_lowercase() == s)
                .mean()
                .alias(f"freq_{s}_spk")
                for s in ("english", "spanish")
            ],
            *[
                (pl.col("race_category").str.to_lowercase() == s)
                .mean()
                .alias(f"freq_{s.split()[0]}")
                for s in ("white", "black or african american", "asian", "other")
            ],
        )
        for ds in dsets
    ]
)

print(
    df_demog.transpose(include_header=True, header_name="stat", column_names="dataset")
    # .to_pandas()
    # .to_latex(index=False, float_format="%.3f")
)

df_tkns = pl.concat(
    [
        pl.read_parquet(hm / "processed" / ds / "tokens_times.parquet").select(
            pl.lit(ds).alias("dataset"),
            pl.len().alias("count"),
            pl.col("tokens").list.len().mean().alias("avg_tkns"),
            pl.col("tokens").list.len().median().alias("med_tkns"),
            (pl.col("times").list.last() - pl.col("times").list.first())
            .mean()
            .alias("avg_los"),
            (pl.col("times").list.last() - pl.col("times").list.first())
            .median()
            .alias("med_los"),
        )
        for ds in dsets
    ]
)

print(df_tkns)
