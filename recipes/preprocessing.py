#!/usr/bin/env python3

"""
load MIMIC, UCMC, & NU;
select patients' first hospitalizations,
restrict to ones >=24h that involve an ICU admission in the first 24h
"""

import os
import pathlib
import shutil

import polars as pl

hm = (
    pathlib.Path("/gpfs/data" if os.uname().nodename.startswith("cri") else "/mnt")
    / "bbj-lab/users/burkh4rt"
)
data_raw = hm / "data-raw"
vers = "3.0.0"


def get_cohort(df_hosp: pl.LazyFrame, df_adt: pl.LazyFrame, h_name: str):
    return (
        df_hosp.drop("__index_level_0__", strict=False)
        .sort(pl.col("admission_dttm"))
        .group_by("patient_id")
        .first()
        .filter(
            pl.col("discharge_dttm") - pl.col("admission_dttm") >= pl.duration(days=1)
        )
        .join(
            df_adt.filter(pl.col("location_category").str.to_lowercase() == "icu")
            .group_by("hospitalization_id")
            .agg(
                pl.col("in_dttm").min().alias("first_icu_admit"),
                pl.col("hospital_id")
                .get(pl.col("in_dttm").arg_min())
                .alias("first_hosp"),
            ),
            on="hospitalization_id",
            validate="1:1",
        )
        .with_columns(
            (
                pl.lit("eicu-" if h == "eicu" else "") + pl.col("first_hosp").cast(str)
            ).alias("first_hosp")
        )
        .filter(
            pl.col("first_icu_admit") <= pl.col("admission_dttm") + pl.duration(days=1)
        )
        .select("patient_id", "hospitalization_id", "first_hosp")
        .cast(str)
    )


def prep_file(f: pathlib.Path, df_cohort: pl.LazyFrame, out_dir: pathlib.Path):
    try:  # hospitalization level
        pl.scan_parquet(f).drop("__index_level_0__", strict=False).cast(
            {"hospitalization_id": str}
        ).join(
            df_cohort.select("hospitalization_id"),
            on="hospitalization_id",
            validate="m:1",
        ).sink_parquet(out_dir / f.name)
        print(f"Processed {f.name} at hospitalization-level.")
    except pl.exceptions.ColumnNotFoundError:  # patient level
        try:
            pl.scan_parquet(f).drop("__index_level_0__", strict=False).cast(
                {"patient_id": str}
            ).join(
                df_cohort.select("patient_id"), on="patient_id", validate="m:1"
            ).sink_parquet(out_dir / f.name)
            print(f"Processed {f.name} at patient-level.")
        except pl.exceptions.ColumnNotFoundError:  # a lookup table
            shutil.copy2(f, out_dir / f.name)
            print(f"Copied {f.name}.")


for h in ("mimic", "ucmc", "nu", "rush", "eicu"):
    df_hosp = pl.scan_parquet(data_raw / f"{h}-{vers}" / "clif_hospitalization.parquet")
    df_adt = pl.scan_parquet(data_raw / f"{h}-{vers}" / "clif_adt.parquet")
    df_cohort = get_cohort(df_hosp, df_adt, h)
    if h in ("nu", "eicu"):
        h_list = sorted(
            df_cohort.group_by("first_hosp")
            .len()
            .filter(pl.col("len") >= 1000)
            .collect()
            .to_series()
            .to_list()
        )
        for hi in h_list:
            (data_raw / f"{hi}-icu-{vers}").mkdir(exist_ok=True)
            for f in (data_raw / f"{h}-{vers}").glob("*.parquet"):
                prep_file(
                    f,
                    df_cohort.filter(pl.col("first_hosp") == pl.lit(hi)),
                    data_raw / f"{hi}-icu-{vers}",
                )
    else:
        (data_raw / f"{h}-icu-{vers}").mkdir(exist_ok=True)
        for f in (data_raw / f"{h}-{vers}").glob("*.parquet"):
            prep_file(f, df_cohort, data_raw / f"{h}-icu-{vers}")
