<p align="center">
<img src="img/coreopsis.png" alt="coreopsis flower" width="400" style="display: block;
margin: 0 auto; -webkit-mask-image: radial-gradient(
    ellipse at center,
    rgba(0,0,0,1) 50%,
    rgba(0,0,0,0) 100%
  );
  mask-image: radial-gradient(
    ellipse at center,
    rgba(0,0,0,1) 50%,
    rgba(0,0,0,0) 100%
  );"/>
</p>

# Coreopsis: choreographed training with flower

[![DOI](img/1306653438.svg)](https://doi.org/10.5281/zenodo.21779190)
[![SWH](https://archive.softwareheritage.org/badge/origin/https://github.com/bbj-lab/coreopsis/)](https://archive.softwareheritage.org/browse/origin/?origin_url=https://github.com/bbj-lab/coreopsis)

> 🌼 over 89 varieties of coreopsis have called Chicago home

## About

This [flower](https://flower.ai) app trains generative event models (GEMs) on
tokenized electronic health records (EHRs) in a federated manner. Each
participating site becomes a client that trains on its own data; only model
weights leave the site, and the server averages them into a shared model once per
round.

In 1989, "the Chicago Botanic Garden created a garden solely to compare
perennials, and _coreopsis_ was one of the inaugural trials." [^1] The
[Lavin Plant Evaluation Garden](https://www.chicagobotanic.org/gardens/planteval)
remains open to this day.

## Installation

```bash
git clone git@github.com:bbj-lab/coreopsis.git
cd coreopsis
mkdir logs
# point `processed/` at the tokenized data produced by cocoa, e.g.
# ln -s ../cocoa/processed ./processed
python -m venv .venv
. .venv/bin/activate
pip install -e . \
  --index-url https://download.pytorch.org/whl/cu128 \
  --extra-index-url https://pypi.org/simple
```

`data-raw/`, `processed/`, and `output/` are typically symlinks to shared
storage; `logs/` needs to exist before the first run.

## Run training

```sh
tmux new -s co || tmux a -t co
. .venv/bin/activate
coreopsis run . | tee "logs/$(date --iso-8601=minutes).stdout"
```

This runs the default (`standard`) federation over every declared dataset for 10
rounds. Logs are not streamed unless `--stream` is passed. Some variations:

```sh
# cpu-only smoke test
coreopsis run . local --stream

# override run parameters (note the nested quoting)
coreopsis run . standard --stream \
  --run-config "'num-server-rounds'=5 'fed-strategy'='FedAdam'"

# a single client, i.e. non-federated training through the same code path
coreopsis run . standard \
  --run-config "'datasets'='[\"mimic-icu\"]'" \
  --federation-config "options.num-supernodes=1"
```

On SLURM, submit [recipes/run_federated.sh](recipes/run_federated.sh), which
wraps the same command and takes its dataset list, strategy, round count, and
output directory from the environment.

### Outputs

After every round the server writes the aggregated model to

```
<output-home>/coreopsis-round-<round>/
```

as a Hugging Face `save_pretrained` directory, so any round of a run can be
passed straight to `cotorra extract` / `cotorra rep-based-score` as a
`--model-home`.

## Configuration

### Flower app (`pyproject.toml`)

The `[tool.flwr.app.config]` table controls top-level training behaviour:

| Key                  | Default                                | Description                                                            |
| -------------------- | -------------------------------------- | ---------------------------------------------------------------------- |
| `datasets`           | `'["mimic-icu","ucmc-icu","nu-icu"]'`  | JSON array of dataset names, one per client partition                  |
| `fed-strategy`       | `"FedAvg"`                             | Federated averaging strategy (`FedAvg`, `FedAvgM`, or `FedAdam`)       |
| `num-server-rounds`  | `10`                                   | Number of federated averaging rounds                                   |
| `output-home`        | `./output/`                            | Directory where checkpoints and the final federated model are saved    |
| `processed-data-dir` | `./processed/`                         | Path to processed data (tokenized timelines, splits, tokenizer config) |
| `training-config`    | `./src/coreopsis/config/training.yaml` | Path to the training configuration YAML [see below]                    |

The server also honours `fraction-fit` and `fraction-evaluate`, but neither is
declared in the table above, so both stay at their `1.0` default — every client
participates in every round — until you add them there.

Federations are defined under `[tool.flwr.federations]`. Three are provided out
of the box:

| Federation             | `num-supernodes` | CPUs per node | GPUs per node |
| ---------------------- | ---------------- | ------------- | ------------- |
| `local`                | 3                | 0.3           | 0             |
| `minimal`              | 3                | 1             | 1             |
| `standard` _(default)_ | 3                | 1             | 1             |

`minimal` and `standard` currently request identical resources; `local` is the
cpu-only configuration used for smoke tests. Run a specific federation with
`coreopsis run . <federation-name>`. Add new federations by adding a
`[tool.flwr.federations.<name>]` block with the same `options.*` keys, or
override the values of an existing one per run with `--federation-config`.

### Collation / tokenization / winnowing

These configurations are borrowed directly from ☕️
[cocoa-tokenizer](https://github.com/bbj-lab/cocoa).

### Training / extraction / scoring

These configurations are borrowed directly from 🦜
[cotorra](https://github.com/bbj-lab/cotorra). Three training configs ship under
[src/coreopsis/config/](src/coreopsis/config/); all describe the same
Llama-3.2-1B–derived architecture (hidden size 1024, 9 layers, 8 heads):

| Config                   | Used for                 | Notes                                                        |
| ------------------------ | ------------------------ | ------------------------------------------------------------ |
| `training.yaml`          | per-site `cotorra train` | 1 epoch; eval/save every 1/100th of training                 |
| `training-no-ckpts.yaml` | federated runs           | eval/save disabled — the server snapshots each round instead |
| `training-star.yaml`     | the `GEM-*` runs         | 5 epochs; eval/save every 1/5th; time-based RoPE and NEFTune |

## Reproducing the experiments

[RUNME.sh](RUNME.sh) is the end-to-end driver used on the cluster: CLIF
harmonization and SOFA scoring, cocoa `collate` → `tokenize` → `winnow`, per-site
`cotorra train`, federated runs sweeping round counts, strategies, and dataset
pairs, then `cotorra extract` → `cotorra rep-based-score` and the tables and
figures produced by [recipes/](recipes/). The `recipes/` scripts are one-off
analysis utilities and are not part of the installed package.

## Modeling ecosystem

This is the federated component of a series of libraries dedicated to
configurable collation and training:

- ☕️ [cocoa-tokenizer](https://pypi.org/project/cocoa-tokenizer/): configurable
  collation and tokenization
- 🦜 [cotorra](https://pypi.org/project/cotorra/): configurable training and
  inference (non-federated)
- 🌼 coreopsis: _this library_

### CLI

We've wrapped the following flower CLI:

```
 Usage: coreopsis [OPTIONS] COMMAND [ARGS]...

 Choreographed federated learning with flower (vXX.X.X)

╭─ Options ───────────────────────────────────────────────────────────────────╮
│ --version             -V        Show the version and exit.                  │
│ --install-completion            Install completion for the current shell.   │
│ --show-completion               Show completion for the current shell, to   │
│                                 copy it or customize the installation.      │
│ --help                -h        Show this message and exit.                 │
╰─────────────────────────────────────────────────────────────────────────────╯
╭─ Commands ──────────────────────────────────────────────────────────────────╮
│ build     Build a Flower App into a Flower App Bundle (FAB).                │
│ install   Install a Flower App Bundle.                                      │
│ log       Get logs from a Flower project run.                               │
│ login     Login to Flower SuperLink.                                        │
│ ls        List the details of one provided run ID or all runs in a Flower   │
│           federation.                                                       │
│ new       Create new Flower App.                                            │
│ run       Run Flower App.                                                   │
│ stop      Stop a run.                                                       │
╰─────────────────────────────────────────────────────────────────────────────╯
```

The primary command to call is `coreopsis run` with documentation as follows:

```
 Usage: coreopsis run [OPTIONS] [APP] [FEDERATION]

 Run Flower App.

╭─ Arguments ─────────────────────────────────────────────────────────────────╮
│   app             [APP]         Path of the Flower App to run. [default: .] │
│   federation      [FEDERATION]  Name of the federation to run the app on.   │
│                                 [default: None]                             │
╰─────────────────────────────────────────────────────────────────────────────╯
╭─ Options ───────────────────────────────────────────────────────────────────╮
│ --run-config         -c      TEXT  Override run configuration values in the │
│                                    format:                                  │
│                                    `--run-config 'key1=value1 key2=value2'  │
│                                    --run-config 'key3=value3'`              │
│                                    Values can be of any type supported in   │
│                                    TOML, such as bool, int, float, or       │
│                                    string. Ensure that the keys (`key1`,    │
│                                    `key2`, `key3` in this example) exist in │
│                                    `pyproject.toml` for proper overriding.  │
│                                    [default: None]                          │
│ --federation-config          TEXT  Override federation configuration values │
│                                    in the format:                           │
│                                    `--federation-config 'key1=value1        │
│                                    key2=value2' --federation-config         │
│                                    'key3=value3'`                           │
│                                    Values can be of any type supported in   │
│                                    TOML, such as bool, int, float, or       │
│                                    string. Ensure that the keys (`key1`,    │
│                                    `key2`, `key3` in this example) exist in │
│                                    the federation configuration under the   │
│                                    `[tool.flwr.federations.<YOUR_FEDERATIO… │
│                                    table of the `pyproject.toml` for proper │
│                                    overriding.                              │
│                                    [default: None]                          │
│ --stream                           Use `--stream` with `flwr run` to        │
│                                    display logs; logs are not streamed by   │
│                                    default.                                 │
│ --format                     TEXT  Format output using 'default' view or    │
│                                    'json'                                   │
│                                    [default: default]                       │
│ --help               -h            Show this message and exit.              │
╰─────────────────────────────────────────────────────────────────────────────╯
```

[^1]:
    R. Hawke, "Coreopsis you can count on!," _Fine Gardening_, No. 171, 44—51,
    https://www.finegardening.com/article/coreopsis-you-can-count-on/

<!--

Run in tmux:
```
tmux new -s co || tmux a -t co
```

Format:
```sh
ruff format .
ruff check . --fix
```

Send to bbj-lab1:
```
for d in data-raw output processed; do
	ln -s /mnt/bbj-lab/users/burkh4rt/$d $d
done
```
```
rsync -avh \
 --exclude "output" \
 --exclude "processed" \
 --exclude "data-raw" \
 --exclude "logs" \
 --exclude "wandb" \
 --exclude ".venv/" \
 --exclude ".idea/" \
 ~/Documents/chicago/coreopsis \
 bbj-lab1:~
```

Send to randi:
```
for d in data-raw output processed; do
	ln -s /gpfs/data/bbj-lab/users/burkh4rt/$d $d
done
```
```
rsync -avh \
 --exclude "output" \
 --exclude "processed" \
 --exclude "data-raw" \
 --exclude "logs" \
 --exclude "wandb" \
 --exclude ".venv/" \
 --exclude ".idea/" \
 ~/Documents/chicago/coreopsis \
 randi:/gpfs/data/bbj-lab/users/burkh4rt
```


Interactive postprocessing:
```
systemd-run --scope --user tmux new -s t3q || tmux a -t t3q
srun -p tier3q \
 --time=8:00:00 \
 --mem=100GB \
 --job-name=adhoc \
 --pty=bash -i
source .venv/bin/activate
```

#SBATCH --partition=bbj-wanq
#SBATCH --qos=bbj-wan_priority
#SBATCH --gres=gpu:1
#SBATCH --time=1-00:00:00

With gpu:
```
systemd-run --scope --user tmux new -s g1 || tmux a -t g1
srun -p bbj-wanq \
 --qos=bbj-wan_priority \
 --gres=gpu:1 \
 --time=8:00:00 \
 --job-name=adhoc \
 --pty bash -i
source .venv/bin/activate
```

-->
