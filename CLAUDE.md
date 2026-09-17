# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with
code in this repository.

## What this is

Coreopsis is the **federated** component of a three-library modeling ecosystem
for training generative event models (GEMs) on tokenized electronic health
records (EHRs):

- ☕️ [cocoa](https://github.com/bbj-lab/cocoa) — configurable collation &
  tokenization
- 🦜 [cotorra](https://github.com/bbj-lab/cotorra) — configurable (non-federated)
  training & inference
- 🌼 coreopsis — _this library_, federated training via
  [Flower](https://flower.ai)

Coreopsis is thin: it wires cotorra's `Trainer` into Flower's client/server apps
and defers all model, collation, tokenization, and scoring logic to cocoa and
cotorra. When debugging model behavior, the code you need is usually in the
installed `cotorra`/`cocoa` packages, not here.

Differential privacy (`TrainerDP`, server-side fixed clipping) was supported
through commit `042f7d2` and removed in `363e5f7`; `cotorra.trainer_dp` still
exists upstream but coreopsis no longer touches it.

## Environment

- `.venv` in the repo root holds the virtualenv with the package installed
  **editable** (`pip install -e .`). Activate with `. .venv/bin/activate`. We do
  **not** use uv.
- To reinstall with the CUDA-pinned torch build:
  ```bash
  pip install -e . --index-url https://download.pytorch.org/whl/cu128 --extra-index-url https://pypi.org/simple
  ```
- `cocoa-tokenizer` and `cotorra` are sibling dependencies, both pinned to exact
  versions in `pyproject.toml`; `clifpy` is pinned to a fork branch. The local
  `.venv` can drift from those pins — check `pip list` before chalking odd
  behavior up to a code change.

## Common commands

```bash
# format + lint (ruff, line-length 88, rules E/F/I, E731 ignored)
ruff format .
ruff check . --fix

# run federated training (the primary entrypoint; wraps `flwr run`)
coreopsis run .                    # default federation ("standard")
coreopsis run . local              # named federation; cpu-only, for smoke tests
coreopsis run . standard --stream --run-config "'num-server-rounds'=10 'datasets'='[\"mimic-icu\"]'"
```

Nothing is printed while a run proceeds unless `--stream` is passed.

There is no test suite. `coreopsis` is a passthrough wrapper around the Flower
CLI (`build`, `install`, `log`, `ls`, `run`, `stop`, …) with a custom help
string; all subcommands are Flower's.

## Configuration lives in two places

1. **`pyproject.toml`** `[tool.flwr.app.config]` — top-level federated run
   parameters: `datasets` (JSON array, one client partition per entry — the
   **single source of truth** for the dataset list, read back outside of
   Flower via `coreopsis.datasets.get_datasets()` /
   `python -m coreopsis.datasets`),
   `fed-strategy` (`FedAvg`/`FedAvgM`/`FedAdam`), `num-server-rounds`, and the
   `processed-data-dir` / `output-home` / `training-config` paths. Federations
   (`local`/`minimal`/`standard`) under `[tool.flwr.federations]` set supernode
   count and CPU/GPU resources (`minimal` and `standard` are currently identical;
   `local` is the cpu-only one). Override any of these at the CLI with
   `--run-config` / `--federation-config`. Two things that bite:
   `options.num-supernodes` must equal `len(datasets)` — clients index into that
   list by `partition-id`, so a mismatch now raises in `FlowerClient.__init__`
   (it used to silently drop trailing datasets when short, `IndexError` when
   long); and the order of `datasets` matters, since the
   server initializes from the **last** entry while RUNME learns the tokenizer
   from the **first**. `server_app.py` also reads `fraction-fit` /
   `fraction-evaluate` from the run config, but neither is declared in the table,
   so both stay at their `1.0` default until you add them there.
2. **`src/coreopsis/config/*.yaml`** — the cotorra/cocoa configs (`collation`,
   `tokenization`, `winnowing`, `training*`, `extraction`, `scoring`). There are
   three training configs; all define the same small Llama-3.2-1B–derived model
   plus HF `training_args`, and all set `custom_loss: false`. Note that the
   `training-config` default in `pyproject.toml` is `training.yaml`, which is
   _not_ what federated runs use — `run_federated.sh` overrides it:
   - `training.yaml` — per-dataset `cotorra train`: 1 epoch, eval/save every
     1/100th of training, `load_best_model_at_end`.
   - `training-no-ckpts.yaml` — federated runs: same model, but eval/save and
     best-model tracking are commented out (the server snapshots each round
     instead).
   - `training-star.yaml` — the `GEM-*` runs: 5 epochs (top-level `n_epochs: 5`,
     which cotorra's `Loader` multiplies against `num_train_epochs: 1`),
     eval/save every 1/5th, plus `time_based_rope` and `neftune_noise_alpha`.

   In cotorra, `time_based_rope` is toggled by the mere **presence** of the key
   (see `Trainer.collate_fn` / `Extractor.collate_fn`). `extraction.yaml` sets
   it, so representations are always extracted with time-derived position ids —
   including for the `c-*` and federated models, which trained with plain
   sequential ones. That matches cotorra's own default `extraction.yaml`, but it
   is a train/extract mismatch for everything except the `GEM-*` runs and is
   worth confirming before interpreting a result.

## Architecture

Flower runs a simulation with one **server** and N **clients** (one per dataset
in `datasets`).

- [src/coreopsis/server_app.py](src/coreopsis/server_app.py) — `server_fn` builds
  the initial model via cotorra's `Trainer.model_init()` (on the **last** dataset
  in `datasets`) and selects a `Save*` strategy by name. `FedAvgM` additionally
  gets a hardcoded `server_learning_rate=1.0` / `server_momentum=0.5`.
- [src/coreopsis/client_app.py](src/coreopsis/client_app.py) — `FlowerClient`
  wraps a cotorra `Trainer`; the client picks its dataset by indexing `datasets`
  with `node_config["partition-id"]`. **A fresh client is created every round.**
  Each round applies a cosine-decayed learning rate and trains on a single
  **shard** of the dataset (`num_shards = num-server-rounds`, `index = round-1`),
  so one full pass over the data is spread across all rounds. Because the
  `Trainer` is rebuilt each round, the decay multiplies the config's
  `learning_rate` rather than compounding. `self.loader` and `self.model` are
  assigned in `__init__` and never used — vestigial, not a hook.
- [src/coreopsis/save_model_strategy.py](src/coreopsis/save_model_strategy.py) —
  `SaveModelMixin` composes with any Flower strategy to snapshot the aggregated
  model after each round, yielding `SaveFedAvg`/`SaveFedAvgM`/`SaveFedAdam`
  (looked up as `Save{fed-strategy}` in the server). Snapshots land in
  `output-home/coreopsis-round-<N>` as HF `save_pretrained` dirs; that name is
  what RUNME's scoring loop consumes as `--model-home`.
- [src/coreopsis/datasets.py](src/coreopsis/datasets.py) — parses the
  `datasets` array out of `pyproject.toml` so the `recipes/` scripts and
  RUNME.sh share one declaration with the Flower app. Inside the client/server
  apps keep using `context.run_config["datasets"]`, which Flower has already
  resolved (including any `--run-config` override).
- [src/coreopsis/task.py](src/coreopsis/task.py) — weight (de)serialization
  to/from numpy (`get_weights`/`set_weights`) and `unpack_context` (resolves
  config/data/output paths from the Flower `Context`).

## End-to-end pipeline

[RUNME.sh](RUNME.sh) is the full driver run on the cluster (SLURM, plus GNU
`parallel`); it is the source of truth for how data flows through the ecosystem.
It is a transcript, not an idempotent script — the `sbatch` steps fan out
asynchronously, so the later stages assume the earlier jobs have finished:

1. `recipes/run_clifpy.py` + `recipes/run_sofa_scoring.py` harmonize raw CLIF
   data and add SOFA scores.
2. `recipes/preprocessing.py` carves the ICU cohort out of `data-raw/<h>-2.1.0`
   into `data-raw/<h>-icu` (each patient's first hospitalization, ≥24h long, with
   an ICU stay starting in the first 24h). Then `cocoa collate` →
   `cocoa tokenize` (tokenizer learned on the first dataset, applied to the rest)
   → `cocoa winnow` → `cocoa combine-datasets` (builds `all`). `winnowing.yaml`
   sets the 24h `threshold`, which is what makes the task "predict the rest of
   the stay from the first day"; the `*_past` / `*_future` columns in
   `*_for_inference.parquet` are that split.
3. Per-dataset training via `cotorra train`, twice: the 1-epoch models with
   `training.yaml` (SLURM, [recipes/run_training.sh](recipes/run_training.sh) →
   `output/c-<dataset>`) and the longer `GEM-*` models with `training-star.yaml`
   (SLURM, [recipes/run_star_training.sh](recipes/run_star_training.sh) →
   `output/cxxx-<dataset>`). RUNME then copies each run's intermediate HF
   `checkpoint-*` dirs out into `output/<run>-NNN/mdl-cotorra`, turning every
   fraction of the data into a standalone model: `c-<site>-001..100` for the
   three sites (where `-100` is not a checkpoint but a copy of the run's final
   `mdl-cotorra`) and `cxxx-<dataset>-001..005` for all four `GEM-*` runs.
   Scoring only reads `{001..010}` and `{015..100..5}` of the `c-*` sweep.
4. Federated training via `coreopsis run` (SLURM,
   [recipes/run_federated.sh](recipes/run_federated.sh), which overrides
   `training-config` to `training-no-ckpts.yaml`). RUNME sweeps
   `num-server-rounds` (1/5/10/50), `fed-strategy`, and the three dataset
   _pairs_, writing `output/c-fedavg10`, `c-fedavg10-{mc,mn,cn}`,
   `c-fed{avgm,adam}10`, etc.
5. `cotorra extract` (representations, into `processed/<ds>/mdl-<run>/`) →
   `cotorra rep-based-score --estimator logistic-CV` → `recipes/baselines.py`
   (LR/LGBM baselines over token presence/counts; writes `bl-*.csv`) →
   `recipes/postprocessing.py` → `recipes/tokenwise.py` (LaTeX tables) →
   `recipes/plotting.py` (plotly PDFs).

`postprocessing.py` writes one bootstrap-CI table per experiment as
`{exp}-{roc,pr}.csv`, where `{exp}` is `tkwz` (per-token), `xfer` (cross-site
transfer), `mthd` (fed strategy), `rnds` (round sweep), or `frac` (data-fraction
sweep plus the leave-one-site-out fed models); the CI cells are numpy `[lo hi]`
strings, and the point estimates in the tables and figures are CI midpoints. The
evaluated outcomes are the tokenizer-vocabulary tokens matching
`tokens_of_interest` in `scoring.yaml` (12 patterns: `RESP//imv`,
`DSCG//expired`, and ten `LABEL//*_init`).

The datasets are whatever `pyproject.toml` declares (plus a combined `all`
built by `cocoa combine-datasets`); as of the `v3` branch that is the two
single-site cohorts, the per-hospital NU splits, `rush`, and the per-hospital
eICU splits, all suffixed with the CLIF version.
`recipes/` scripts are one-off analysis/plotting utilities, not part of the
installed package. Only `demographics.py` is **not** wired into RUNME.sh. Note
the path split: `postprocessing.py`, `baselines.py`, and `demographics.py` read
and write the cluster share (`/gpfs/data` or `/mnt` + `bbj-lab/users/burkh4rt`),
while `tokenwise.py` and `plotting.py` resolve everything under `~/Downloads`.
RUNME.sh runs `baselines` → `postprocessing` → `tokenwise` → `plotting` back to
back, so the last two only work once their inputs have been copied down — in
practice the tables and figures get made locally.

## Conventions

- Data directories (`data-raw/`, `processed/`, `output/`, `logs/`, `wandb/`) are
  typically symlinks to shared cluster storage; create `logs/` before running.
  `.gitignore` covers `processed/`, `output/`, and `wandb/` but **not**
  `data-raw/` or `logs/` (only `*.log` inside it, so the SLURM `*.stdout` files
  are untracked-but-visible).
- Weights & Biases logging is currently **off**: every training config sets
  `report_to: none` and all three SLURM scripts export `WANDB_MODE=offline`. The
  project name to use when re-enabling it is `coreopsis`.
- The SLURM scripts also export `HF_HUB_OFFLINE=1`, so the
  `meta-llama/Llama-3.2-1B` config the models derive from has to already be in
  the local HF cache on the compute node.
- toml/yaml/markdown formatting via taplo/prettier (`.taplo.toml`,
  `.prettierrc.toml`; prose wraps at 81 columns, `proseWrap = "always"`, but yaml
  at 100). Editor defaults live in `.vscode/settings.json`.
