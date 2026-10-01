import matplotlib
import numpy as np
import polars as pl

matplotlib.use("Agg")
import matplotlib.pyplot as plt

TOKS = {
    "RESP//imv": 1193,
    "DSCG//expired": 117,
    "LABEL//anemia_init": 648,
    "LABEL//crrt_init": 649,
    "LABEL//hyperk_init": 650,
    "LABEL//hypok_init": 653,
    "LABEL//hyperna_init": 651,
    "LABEL//hypona_init": 654,
    "LABEL//hypertension_init": 652,
    "LABEL//hypotension_init": 655,
    "LABEL//pressor_init": 656,
    "LABEL//tachy_init": 658,
}
df = pl.read_parquet("/Users/michael/Downloads/held_out_for_inference.parquet")
res = {}
long = []
for name, t in TOKS.items():
    d = (
        df.filter(pl.col("tokens").list.contains(t))
        .with_columns(
            i=pl.col("tokens").list.eval(pl.element().index_of(t)).list.first()
        )
        .select(
            "subject_id",
            hours=pl.col("s_elapsed").list.get(pl.col("i")) / 3600,
            past=pl.col(f"{name}_past"),
        )
    )
    h = d["hours"].to_numpy()
    assert ((h < 24) == d["past"].to_numpy()).all(), name
    res[name] = h
    long.append(d.select("subject_id", pl.lit(name).alias("outcome"), "hours"))
    print(
        f"{name:26s} n={len(h):5d}  median={np.median(h):7.1f}h"
        f"  <24h={np.mean(h < 24):4.0%}  24-48h={np.mean((h >= 24) & (h < 48)):4.0%}"
        f"  max={h.max():8.0f}h"
    )
pl.concat(long).rename({"hours": "hours_from_admission"}).write_csv(
    "/Users/michael/Downloads/outcome_times.csv"
)

S, T1, T2, C = "#fcfcfb", "#0b0b0b", "#52514e", "#2a78d6"
allh = np.concatenate(list(res.values()))
bins = np.logspace(
    np.log10(max(allh.min(), 1 / 60) * 0.9), np.log10(allh.max() * 1.1), 50
)
ticks = [1 / 6, 1, 6, 24, 168, 720, 8760]
tl = ["10m", "1h", "6h", "1d", "1wk", "30d", "1y"]
fig, axes = plt.subplots(4, 3, figsize=(12, 11), sharex=True, facecolor=S)
for ax, (name, h) in zip(axes.flat, res.items()):
    ax.set_facecolor(S)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(T2)
    ax.tick_params(colors=T2, labelsize=8)
    ax.grid(axis="y", color="#e8e7e3", lw=0.8)
    ax.set_axisbelow(True)
    ax.hist(
        h,
        bins=bins,
        weights=np.full(len(h), 100 / len(h)),
        color=C,
        edgecolor=S,
        linewidth=0.6,
    )
    ax.axvline(24, color=T1, lw=1.5, ls=(0, (5, 3)))
    ax.axvline(48, color=T1, lw=1.5, ls=(0, (1, 2)))
    ax.set_xscale("log")
    ax.set_title(name, color=T1, fontsize=10, loc="left")
    ax.text(
        0.98,
        0.95,
        f"n = {len(h):,}\nmedian {np.median(h):.0f}h\n"
        f"{np.mean(h < 24):.0%} < 24h\n{np.mean((h >= 24) & (h < 48)):.0%} 24–48h",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        color=T2,
    )
    ax.set_xticks(ticks, tl)
for ax in axes[:, 0]:
    ax.set_ylabel("% of patients", color=T1, fontsize=9)
for ax in axes[-1]:
    ax.set_xlabel("Time from admission (log)", color=T1, fontsize=9)
fig.suptitle(
    "Time from admission to first occurrence of each outcome token  "
    "(dashed: admission + 24h, dotted: + 48h)",
    color=T1,
    x=0.01,
    ha="left",
    fontsize=12,
)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(
        f"/Users/michael/Downloads/outcome_times_from_admission.{ext}",
        dpi=200,
        facecolor=S,
    )
