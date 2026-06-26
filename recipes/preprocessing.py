#!/usr/bin/env python3

"""
load MIMIC / UCMC and split first hospitalizations by date
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
partition MIMIC by anchor year
"""

mimic_physionet = hm / "physionet.org" / "files" / "mimiciv" / "3.1"
df = (
    pl.read_csv(mimic_physionet / "hosp" / "patients.csv.gz")
    .join(
        pl.read_csv(mimic_physionet / "hosp" / "admissions.csv.gz")
        .sort(pl.col("admittime").str.strptime(pl.Datetime, strict=False))
        .group_by("subject_id")
        .first(),
        on="subject_id",
        validate="1:1",
    )
    .cast(str)
)
for k, v in {
    "08": "2008 - 2010",
    "11": "2011 - 2013",
    "14": "2014 - 2016",
    "17": "2017 - 2019",
    "20": "2020 - 2022",
}.items():
    (data_raw / f"mimic-{k}").mkdir(exist_ok=True)
    grp = df.filter(pl.col("anchor_year_group") == v).select(
        pl.col("subject_id").alias("patient_id"),
        pl.col("hadm_id").alias("hospitalization_id"),
    )
    for f in (data_raw / "mimic-2.1.0").glob("*.parquet"):
        try:
            pl.scan_parquet(f).join(
                grp.select("hospitalization_id").lazy(),
                on="hospitalization_id",
                validate="m:1",
            ).sink_parquet(data_raw / f"mimic-{k}" / f.name)
        except pl.exceptions.ColumnNotFoundError:
            pl.scan_parquet(f).join(
                grp.select("patient_id").lazy(), on="patient_id", validate="m:1"
            ).sink_parquet(data_raw / f"mimic-{k}" / f.name)


"""
partition UCMC & Northwestern by admission year
"""

for h in ("ucmc", "nu"):
    df = (
        pl.read_parquet(data_raw / f"{h}-2.1.0" / "clif_hospitalization.parquet")
        .sort(pl.col("admission_dttm"))
        .group_by("patient_id")
        .first()
    )
    for k, v in {
        "18": 2018,
        "19": 2019,
        "20": 2020,
        "21": 2021,
        "22": 2022,
        "23": 2023,
        "24": 2024,
    }.items():
        (data_raw / f"{h}-{k}").mkdir(exist_ok=True)
        grp = df.filter(pl.col("admission_dttm").dt.year() == v)
        for f in (data_raw / f"{h}-2.1.0").glob("*.parquet"):
            try:  # hospitalization level
                pl.scan_parquet(f).cast({"hospitalization_id": str}).join(
                    grp.select("hospitalization_id").lazy(),
                    on="hospitalization_id",
                    validate="m:1",
                ).sink_parquet(data_raw / f"{h}-{k}" / f.name)
                print(f"Processed {f.name} at hospitalizion-level.")
            except pl.exceptions.ColumnNotFoundError:  # patient level
                try:
                    pl.scan_parquet(f).join(
                        grp.select("patient_id").lazy(), on="patient_id", validate="m:1"
                    ).sink_parquet(data_raw / f"{h}-{k}" / f.name)
                    print(f"Processed {f.name} at patient-level.")
                except pl.exceptions.ColumnNotFoundError:  # a lookup table
                    shutil.copy2(f, data_raw / f"{h}-{k}" / f.name)
                    print(f"Copied {f.name}.")
