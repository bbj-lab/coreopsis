#!/usr/bin/env python3

"""
load MIMIC and split hospitalizations by date (<2014, >=2014)
"""

import pathlib

import polars as pl

hm = pathlib.Path("/gpfs/data/bbj-lab/users/burkh4rt")
data_raw = hm / "data-raw"
mimic_clif = data_raw / "mimic-2.1.0"
mimic_physionet = hm / "physionet.org" / "files" / "mimiciv" / "3.1"

df_patients = pl.read_csv(mimic_physionet / "hosp" / "patients.csv.gz")

cond = pl.col("anchor_year_group").is_in(["2008 - 2010", "2011 - 2013"])

hosp = (
    pl.read_csv(mimic_physionet / "hosp" / "admissions.csv.gz")
    .sort(pl.col("admittime").str.strptime(pl.Datetime, strict=False))
    .group_by("subject_id")
    .first()
)

hosp_pre14 = (
    hosp.join(
        df_patients.filter(cond).select("subject_id"), on="subject_id", how="inner"
    )
    .select(pl.col("hadm_id").cast(pl.String).alias("hospitalization_id"))
    .lazy()
)
hosp_post14 = (
    hosp.join(
        df_patients.filter(~cond).select("subject_id"), on="subject_id", how="inner"
    )
    .select(pl.col("hadm_id").cast(pl.String).alias("hospitalization_id"))
    .lazy()
)

pl.scan_parquet(
    h_pre14 := data_raw / "mimic-pre14" / "clif_hospitalization.parquet"
).join(hosp_pre14, on="hospitalization_id", how="inner").sink_parquet(h_pre14)
pl.scan_parquet(
    h_post14 := data_raw / "mimic-post14" / "clif_hospitalization.parquet"
).join(hosp_post14, on="hospitalization_id", how="inner").sink_parquet(h_post14)

(
    pl.scan_parquet(data_raw / "ucmc-2.1.0" / "clif_hospitalization.parquet")
    .sort("admission_dttm")
    .group_by("patient_id")
    .first()
    .sink_parquet(data_raw / "ucmc-first" / "clif_hospitalization_first.parquet")
)
