<p align="center">
<img src="img/coreopsis.png" alt="cocoa bean" width="400" style="display: block;
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

> 🌼 over 89 varieties of coreopsis have called Chicago home

## About

This [flower](https://flower.ai) app trains generative event models (GEMs) on
tokenized electronic health records (EHRs) in a federated manner. In 1989, "the
Chicago Botanic Garden created a garden solely to compare perennials, and
_coreopsis_ was one of the inaugural trials." [^1] The
[Lavin Plant Evaluation Garden](https://www.chicagobotanic.org/gardens/planteval)
remains open to this day.

## Installation

```bash
git clone git@github.com:bbj-lab/coreopsis.git
cd coreopsis
# ln -s ../cocoa/processed ./processed
mkdir logs
python -m venv .venv
. .venv/bin/activate
pip install -e . \
  --index-url https://download.pytorch.org/whl/cu128 \
  --extra-index-url https://pypi.org/simple
```

## Run training

```sh
tmux new -s co || tmux a -t co
. .venv/bin/activate
coreopsis run . | tee "logs/$(date --iso-8601=minutes).stddout"
```

## Configuration

### Flower app (`pyproject.toml`)

The `[tool.flwr.app.config]` table controls top-level training behaviour:

| Key                  | Default                                | Description                                                             |
| -------------------- | -------------------------------------- | ----------------------------------------------------------------------- |
| `datasets`           | `'["mimic-icu","ucmc-icu","nu-icu"]'`  | JSON array of dataset names, one per client partition                   |
| `diff-priv-client`   | `0`                                    | Enable differential privacy on the client (`1` for true, `0` for false) |
| `diff-priv-server`   | `0`                                    | Enable differential privacy on the server (`1` for true, `0` for false) |
| `fed-strategy`       | `"FedAvg"`                             | Federated averaging strategy (`FedAvg`, `FedAvgM`, or `FedAdam`)        |
| `max-grad-norm`      | `1.0`                                  | Maximum gradient norm for clipping (used with differential privacy)     |
| `noise-multiplier`   | `1.5`                                  | Noise multiplier for differential privacy                               |
| `num-server-rounds`  | `10`                                   | Number of federated averaging rounds                                    |
| `output-home`        | `./output/`                            | Directory where checkpoints and the final federated model are saved     |
| `processed-data-dir` | `./processed/`                         | Path to processed data (tokenized timelines, splits, tokenizer config)  |
| `training-config`    | `./src/coreopsis/config/training.yaml` | Path to the training configuration YAML [see below]                     |

Federations are defined under `[tool.flwr.federations]`. Three are provided out
of the box:

| Federation             | `num-supernodes` | CPUs per node | GPUs per node |
| ---------------------- | ---------------- | ------------- | ------------- |
| `local`                | 3                | 0.3           | 0             |
| `minimal`              | 3                | 1             | 1             |
| `standard` _(default)_ | 3                | 1             | 1             |

Run a specific federation with `coreopsis run . <federation-name>`. Add new
federations by adding a `[tool.flwr.federations.<name>]` block with the same
`options.*` keys.

### Collation / tokenization / winnowing

These configurations are borrowed directly from ☕️
[cocoa](https://github.com/bbj-lab/cocoa).

### Training / extraction / scoring

These configurations are borrowed directly from 🦜
[cotorra](https://github.com/bbj-lab/cotorra).

## Modeling ecosystem

This is the federated component of a series of libraries dedicated to
configurable collation and training:

- ☕️ [cocoa](https://github.com/bbj-lab/cocoa): configurable collation and
  tokenization
- 🦜 [cotorra](https://github.com/bbj-lab/cotorra): configurable training and
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

rsync -avh \
 bbj-lab1:~/coreopsis/processed \
 ~/Downloads/

rsync -avh \
 ~/Downloads/processed \
 randi:/gpfs/data/bbj-lab/users/burkh4rt/

-->
