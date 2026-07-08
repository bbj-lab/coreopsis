#!/usr/bin/env python3

"""
Run sofa scoring pipeline
"""

import argparse
import pathlib

import clifpy as cp

parser = argparse.ArgumentParser()
parser.add_argument(
    "--data_dir", type=pathlib.Path, default="../../data-raw/ucmc-2.1.0"
)
parser.add_argument("--out_dir", type=pathlib.Path, default="../../figs")
parser.add_argument("--tz", type=str, default="UTC")
args, unknowns = parser.parse_known_args()

data_dir, out_dir = map(
    lambda x: pathlib.Path(x).expanduser().resolve(), (args.data_dir, args.out_dir)
)

co = cp.ClifOrchestrator(
    data_directory=str(data_dir),
    filetype="parquet",
    timezone=args.tz,
    output_directory=str(out_dir),
)

co.convert_dose_units_for_continuous_meds(
    preferred_units={
        "norepinephrine": "mcg/kg/min",
        "epinephrine": "mcg/kg/min",
        "dopamine": "mcg/kg/min",
        "dobutamine": "mcg/kg/min",
    },
    override=True,
)

sofa_scores = co.compute_sofa_scores()

sofa_df = (
    co.wide_df.sort_values("event_time")
    .groupby("encounter_block")
    .agg({"event_time": "max", "hospitalization_id": "last"})
    .join(co.sofa_df, how="inner", on="encounter_block", validate="1:1")
)

sofa_df.to_parquet(out_loc := data_dir / "clif_sofa.parquet")
