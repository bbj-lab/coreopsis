#!/bin/bash

#SBATCH --job-name=coreopsis
#SBATCH --output=./logs/training-%j.stdout
#SBATCH --partition=gpuq
#SBATCH --qos=priority
#SBATCH --gres=gpu:4
#SBATCH --time=12:00:00

export HF_HUB_OFFLINE=1
export WANDB_MODE=offline

source ~/.bashrc
source .venv/bin/activate

coreopsis run . standard \
	--stream \
	--run-config "
				 'fed-strategy'='${fed_strategy:-FedAvg}'
				 'output-home'='${output_home:-./output/c-fedavg10}'
				 'num-server-rounds'=${num_server_rounds:-10}
				 'datasets'='[$dsets_cfg]'
				 'training-config'='./src/coreopsis/config/training-no-ckpts.yaml'
				 " \
	--federation-config "
						options.num-supernodes=${nsets}
						options.backend.client-resources.num-cpus=1
						options.backend.client-resources.num-gpus=1
						"
