#!/usr/bin/env python3

"""
Run the CLIFpy package on CLIF-2.1 datasets
"""

import argparse
import pathlib

import clifpy as cpy
from clifpy.tables.respiratory_support import RespiratorySupport

parser = argparse.ArgumentParser()
parser.add_argument(
    "--data_dir", type=pathlib.Path, default="../../data-raw/mimic-2.1.0"
)
parser.add_argument("--out_dir", type=pathlib.Path, default="../../figs")
parser.add_argument("--tz", type=str, default="UTC")
parser.add_argument(
    "--tables",
    type=str,
    nargs="*",
    default=[
        "patient",
        "hospitalization",
        "adt",
        "labs",
        "vitals",
        "medication_admin_continuous",
        "medication_admin_intermittent",
        "patient_assessments",
        "respiratory_support",
        "patient_procedures",
        "code_status",
    ],
)
parser.add_argument("--validate", action="store_true")
parser.add_argument("--waterfall", action="store_true")
parser.add_argument("--convert_doses_continuous", action="store_true")
parser.add_argument("--convert_doses_intermittent", action="store_true")
args, unknowns = parser.parse_known_args()

data_dir, out_dir = map(
    lambda x: pathlib.Path(x).expanduser().resolve(), (args.data_dir, args.out_dir)
)

co = cpy.ClifOrchestrator(
    data_directory=str(data_dir),
    filetype="parquet",
    timezone=args.tz,
    output_directory=str(out_dir),
)

if args.validate:
    co.initialize(tables=args.tables)
    co.validate_all()

if args.waterfall:
    resp_support = RespiratorySupport.from_file(
        data_directory=str(data_dir), filetype="parquet", timezone=args.tz
    )
    processed = resp_support.waterfall()
    processed.validate()
    processed.df.to_parquet(data_dir / "clif_respiratory_support_processed.parquet")


if args.convert_doses_continuous:
    # opts = {"data_directory": str(data_dir), "timezone": args.tz, "filetype": "parquet"}
    # mac_df = cpy.MedicationAdminContinuous.from_file(**opts).df
    #
    # preferred_units = dict(
    #     pl.from_pandas(mac_df)
    #     .group_by("med_category")
    #     .agg(pl.col("med_dose_unit").mode().first())
    #     .sort("med_category")
    #     .rows()
    # )

    preferred_units = {
        "acetaminophen": "mg/min",
        "albumin_infusion": "ml/hr",
        "alteplase": "mg/hr",
        "aminocaproic": "g/hr",
        "aminophylline": "mg/kg/hr",
        "amiodarone": "mg/min",
        "angiotensin": "ng/kg/min",
        "argatroban": "mcg/kg/min",
        "bivalirudin": "mg/kg/hr",
        "bumetanide": "mg/hr",
        "cisatracurium": "mg/kg/hr",
        "clevidipine": "mg/hr",
        "dexmedetomidine": "mcg/kg/hr",
        "dextrose": "ml/hr",
        "dextrose_in_water_d5w": "ml/hr",
        "diltiazem": "mg/hr",
        "dobutamine": "mcg/kg/min",
        "dopamine": "mcg/kg/min",
        "epinephrine": "mcg/kg/min",
        "epoprostenol": "ng/kg/min",
        "eptifibatide": "mcg/kg/min",
        "esmolol": "mcg/kg/min",
        "fentanyl": "mcg/hr",
        "furosemide": "mg/hr",
        "heparin": "u/hr",
        "hydromorphone": "mg/hr",
        "insulin": "u/hr",
        "ketamine": "mg/kg/hr",
        "labetalol": "mg/min",
        "lidocaine": "mg/min",
        "lorazepam": "mg/hr",
        "magnesium": "g/hr",
        "midazolam": "mg/hr",
        "milrinone": "mcg/kg/min",
        "morphine": "mg/hr",
        "naloxone": "mg/hr",
        "nicardipine": "mcg/kg/min",
        "nitroglycerin": "mcg/kg/min",
        "nitroprusside": "mcg/kg/min",
        "norepinephrine": "mcg/kg/min",
        "octreotide": "mcg/hr",
        "pantoprazole": "mg/hr",
        "pentobarbital": "mg/kg/hr",
        "phenylephrine": "mcg/kg/min",
        "procainamide": "mg/min",
        "propofol": "mcg/kg/min",
        "rocuronium": "mcg/kg/min",
        "sodium chloride": "ml/hr",
        "tpn": "ml/hr",
        "treprostinil": "ng/kg/min",
        "vasopressin": "u/hr",
        "vecuronium": "mg/kg/hr",
    }

    co.convert_dose_units_for_continuous_meds(
        preferred_units=preferred_units,
        show_intermediate=True,
        override=True,
        save_to_table=True,
    )

    co.medication_admin_continuous.df_converted.to_parquet(
        data_dir / "clif_medication_admin_continuous_converted.parquet"
    )

if args.convert_doses_intermittent:
    # opts = {"data_directory": str(data_dir), "timezone": args.tz, "filetype": "parquet"}
    # mai_df = cpy.MedicationAdminIntermittent.from_file(**opts).df
    #
    # preferred_units = dict(
    #     pl.from_pandas(mai_df)
    #     .group_by("med_category")
    #     .agg(pl.col("med_dose_unit").mode().first())
    #     .sort("med_category")
    #     .rows()
    # )

    preferred_units = {
        "acetaminophen": "mg",
        "acyclovir": "dose",
        "adenosine": "mg",
        "amikacin": "dose",
        "amiodarone": "mg",
        "ampicillin": "dose",
        "ampicillin_sulbactam": "dose",
        "azithromycin": "dose",
        "aztreonam": "dose",
        "bumetanide": "mg",
        "caspofungin": "dose",
        "cefazolin": "dose",
        "cefepime": "dose",
        "ceftaroline": "dose",
        "ceftazidime": "dose",
        "ceftriaxone": "dose",
        "ciprofloxacin": "dose",
        "cisatracurium": "mg",
        "clindamycin": "dose",
        "colistin": "dose",
        "daptomycin": "dose",
        "dextrose": "ml",
        "dextrose_in_water_d5w": "ml",
        "diazepam": "mg",
        "diltiazem": "mg",
        "doxycycline": "dose",
        "epinephrine": "mg",
        "ertapenem": "dose",
        "erythromycin": "dose",
        "esomeprazole": "dose",
        "fentanyl": "mcg",
        "fluconazole": "dose",
        "foscarnet": "dose",
        "furosemide": "mg",
        "gentamicin": "dose",
        "heparin": "dose",
        "hydromorphone": "mg",
        "imipenem": "dose",
        "insulin": "units",
        "ketamine": "mcg",
        "labetalol": "mg",
        "levofloxacin": "dose",
        "lidocaine": "mg",
        "linezolid": "dose",
        "lorazepam": "mg",
        "magnesium": "grams",
        "meropenem": "dose",
        "metronidazole": "dose",
        "micafungin": "dose",
        "midazolam": "mg",
        "morphine": "mg",
        "moxifloxacin": "dose",
        "nafcillin": "dose",
        "naloxone": "mg",
        "pantoprazole": "dose",
        "penicillin": "dose",
        "piperacillin_tazobactam": "dose",
        "propofol": "mg",
        "rifampin": "dose",
        "rocuronium": "mg",
        "sodium bicarbonate": "ml",
        "sodium chloride": "ml",
        "tigecycline": "dose",
        "tobramycin": "dose",
        "vancomycin": "dose",
        "vecuronium": "mg",
        "voriconazole": "dose",
    }

    co.convert_dose_units_for_intermittent_meds(
        preferred_units=preferred_units,
        show_intermediate=True,
        override=True,
        save_to_table=True,
    )

    co.medication_admin_intermittent.df_converted.to_parquet(
        data_dir / "clif_medication_admin_intermittent_converted.parquet"
    )
