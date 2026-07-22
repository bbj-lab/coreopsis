#!/usr/bin/env python3

"""
load and select tabular results
"""

import pathlib

import pandas as pd

pd.options.display.float_format = "{:,.3f}".format
pd.options.display.max_columns = None
pd.options.display.max_rows = 100
pd.options.display.width = None
pd.options.display.expand_frame_repr = False
pd.options.display.show_dimensions = True

hm = pathlib.Path("~/Downloads").expanduser().resolve()
dsets = ("ucmc-icu", "nu-icu", "mimic-icu")
methods = ("fedavg", "fedavgm", "fedadam")


def fmt_ci(s, p=3):
    """Format a stringified CI (e.g. "[0.76768137 0.79238952]") to p decimals."""
    lo, hi = (float(v) for v in s.strip("[]").split())
    return f"[{lo:.{p}f}, {hi:.{p}f}]"

agg_roc_ci = pd.read_csv(hm / "aggregate-roc-cis.csv", index_col=0).rename_axis(
    "models"
)
agg_pr_ci = pd.read_csv(hm / "aggregate-pr-cis.csv", index_col=0).rename_axis("models")

tks_roc = pd.read_csv(hm / "tokenwise-roc-auc.csv").set_index(["token", "models"])
tks_pr = pd.read_csv(hm / "tokenwise-pr-auc.csv").set_index(["token", "models"])

agg_roc = tks_roc.groupby("models").mean()
agg_pr = tks_pr.groupby("models").mean()

# transfer
mdls = (
    [f"mdl-{ds}-100" for ds in dsets]
    + [f"mdl-{method}10" for method in methods]
    + ["mdl-all"]
)
print(agg_roc_ci.loc[mdls].map(fmt_ci))
print(agg_pr_ci.loc[mdls].map(fmt_ci))
