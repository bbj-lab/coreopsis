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

"""
partition UCMC & Northwestern by admission year
"""

for h in ("mimic", "ucmc", "nu"):
    cohort = (
        pl.read_parquet(data_raw / f"{h}-2.1.0" / "clif_hospitalization.parquet")
        .drop("__index_level_0__", strict=False)
        .sort(pl.col("admission_dttm"))
        .group_by("patient_id")
        .first()
        .filter(
            pl.col("discharge_dttm") - pl.col("admission_dttm") >= pl.duration(days=1)
        )
        .join(
            pl.read_parquet(data_raw / f"{h}-2.1.0" / "clif_adt.parquet")
            .filter(pl.col("location_category").str.to_lowercase() == "icu")
            .group_by("hospitalization_id")
            .agg(pl.col("in_dttm").min().alias("first_icu_admit")),
            on="hospitalization_id",
            validate="1:1",
        )
        .filter(
            pl.col("first_icu_admit") <= pl.col("admission_dttm") + pl.duration(days=1)
        )
        .select("patient_id", "hospitalization_id")
        .lazy()
    )
    (data_raw / f"{h}-icu").mkdir(exist_ok=True)
    for f in (data_raw / f"{h}-2.1.0").glob("*.parquet"):
        try:  # hospitalization level
            pl.scan_parquet(f).drop("__index_level_0__", strict=False).cast(
                {"hospitalization_id": str}
            ).join(
                cohort.select("hospitalization_id"),
                on="hospitalization_id",
                validate="m:1",
            ).sink_parquet(data_raw / f"{h}-icu" / f.name)
            print(f"Processed {f.name} at hospitalizion-level.")
        except pl.exceptions.ColumnNotFoundError:  # patient level
            try:
                pl.scan_parquet(f).drop("__index_level_0__", strict=False).join(
                    cohort.select("patient_id"), on="patient_id", validate="m:1"
                ).sink_parquet(data_raw / f"{h}-icu" / f.name)
                print(f"Processed {f.name} at patient-level.")
            except pl.exceptions.ColumnNotFoundError:  # a lookup table
                shutil.copy2(f, data_raw / f"{h}-icu" / f.name)
                print(f"Copied {f.name}.")
