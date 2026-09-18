#!/bin/bash

#SBATCH --job-name=coreopsis
#SBATCH --output=./logs/training-%j.stdout
#SBATCH --partition=bbjq
#SBATCH --qos=bbj_priority
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00

export HF_HUB_OFFLINE=1
export WANDB_MODE=offline

source ~/.bashrc
source .venv/bin/activate

cotorra train \
	--training-config ${config_home}/training.yaml \
	--processed-data-home ./processed/${ds} \
	--output-home ./output/c-${ds}
