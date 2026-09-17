#!/usr/bin/env python3

"""
gather stats
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

dsets = os.getenv("dsets_csv").strip(",").split(",") + ["all"]

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
            + [
                pl.col(f"{c}_future").mean().alias(c.split("//")[-1] + "_future")
                for c in labels
            ]
        )
        for ds in dsets
    ]
)

print(
    df_outcomes
    # .to_pandas()
    # .to_latex(index=False, float_format="%.3f")
)

languages = ["english", "spanish"]
races = ["white", "black_or_african_american", "asian"]
df_demog = pl.concat(
    [
        pl.read_parquet(
            list((hm / "processed" / ds).glob("*_for_inference.parquet"))
        ).select(
            pl.lit(ds).str.split("-icu-").list[0].alias("dataset"),
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
                (pl.col("race_category").str.to_lowercase() == s)
                .mean()
                .alias(f"freq_{s.split()[0]}")
                for s in races
            ],
            (~pl.col("race_category").str.to_lowercase().is_in(races))
            .mean()
            .alias("freq_other"),
            *[
                (pl.col("language_category").cast(str).str.to_lowercase() == s)
                .mean()
                .alias(f"freq_{s}_spk")
                for s in languages
            ],
            (
                (
                    ~pl.col("language_category")
                    .cast(str)
                    .str.to_lowercase()
                    .is_in(languages)
                )
                .mean()
                .alias("freq_other_spk")
            ),
        )
        for ds in dsets
    ]
)

# .transpose(include_header=True, header_name="stat", column_names="dataset")
print(df_demog.to_pandas().to_latex(index=False, float_format="%.3f"))

df_tkns = pl.concat(
    [
        pl.read_parquet(hm / "processed" / ds / "tokens_times.parquet").select(
            pl.lit(ds).str.split("-icu-").list[0].alias("dataset"),
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


vocab = OmegaConf.load(hm / "processed" / dsets[0] / "tokenizer.yaml").lookup.keys()
token_types = (
    pl.Series(name="vocab", values=vocab)
    .str.split("//")
    .list.first()
    .value_counts()
    .sort("vocab")
)

df_training = {
    ds: pl.read_parquet(hm / "processed" / ds / "train_tokens_times.parquet")
    for ds in dsets
}

print(
    {
        k: v.select(pl.col("tokens").list.len().sum()).item()
        for k, v in df_training.items()
    }
)

print({k: v.shape[0] for k, v in df_training.items()})

print(
    {
        k: v.select(
            (pl.col("times").list.max() - pl.col("times").list.min())
            .dt.total_days()
            .alias("duration")
            .mean()
        ).item()
        for k, v in df_training.items()
    }
)

df_test = {
    ds: pl.read_parquet(hm / "processed" / ds / "held_out_for_inference.parquet")
    for ds in dsets
}

print(
    {
        k: v.select(pl.col("tokens_past").list.len().mean()).item()
        for k, v in df_test.items()
    }
)

print({k: v.shape[0] for k, v in df_test.items()})


rev = {
    v: k
    for k, v in OmegaConf.load(hm / "processed" / "ucmc-icu" / "tokenizer.yaml")[
        "lookup"
    ].items()
}

df = (
    pl.read_parquet(hm / "processed" / "ucmc-icu" / "held_out_for_inference.parquet")
    .filter("RESP//imv_past")
    .with_columns(pl.col("tokens").list.eval(pl.element().replace_strict(rev)))
)

", ".join(
    [
        f"<{x}>"
        for x in (
            df.filter(pl.col("subject_id") == "642969984").select("tokens").item()
        )
    ]
)


df = (
    pl.read_parquet(hm / "processed" / "ucmc-icu" / "held_out_for_inference.parquet")
    .filter("LABEL//pressor_init_past")
    .with_columns(pl.col("tokens").list.eval(pl.element().replace_strict(rev)))
)
