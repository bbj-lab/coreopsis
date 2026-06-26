#!/bin/bash

# tmux new -s co || tmux a -t co
# ln -s /mnt/bbj-lab/users/burkh4rt/data-raw ./data-raw

. .venv/bin/activate
python3 preprocessing.py

dsets=(mimic-{08..20..3} ucmc-{18..24})
dsets_cfg=$(printf '"%s",' "${dsets[@]}")
dsets_cfg=${dsets_cfg%,}
config_home=./src/coreopsis/config

# collate data
parallel --bar -j 4 cocoa collate \
	--collation-config ${config_home}/collation.yaml \
	--raw-data-home ./data-raw/{} \
	--processed-data-home ./processed/{} \
	::: "${dsets[@]}"

# learn tokenizer on first dataset
cocoa tokenize \
	--tokenization-config ${config_home}/tokenization.yaml \
	--processed-data-home ./processed/${dsets[0]}

# apply tokenizer to other datasets
parallel --bar -j 4 cocoa tokenize \
	--tokenization-config ${config_home}/tokenization.yaml \
	--tokenizer-home ./processed/${dsets[0]}/tokenizer.yaml \
	--processed-data-home ./processed/{} \
	::: "${dsets[@]:1}"

# winnow data (prepare for inference)
parallel --bar -j 4 cocoa winnow \
	--winnowing-config ${config_home}/winnowing.yaml \
	--processed-data-home ./processed/{} \
	::: "${dsets[@]}"

# create a combined dataset
cocoa combine-datasets \
	"${dsets[@]/#/./processed/}" \
	--output-data-dir ./processed/all

dsets+=('all')

# train separate models on each dataset
for ds in "${dsets[@]}"; do
	cotorra train \
		--training-config ${config_home}/training.yaml \
		--processed-data-home ./processed/${ds} \
		--output-home ./output/${ds} \
		2>&1 | tee ./logs/training-${ds}.log
done

# train separate models on each dataset
for ds in "${dsets[@]}"; do
	cotorra train-private \
		--training-config ${config_home}/training.yaml \
		--processed-data-home ./processed/${ds} \
		--output-home ./output/${ds}-p \
		2>&1 | tee ./logs/training-${ds}-p.log
done

# run federated learning
coreopsis run . standard \
	--stream \
	--run-config "
				 'fed-strategy'='FedAvg'
				 'output-home'='./output/fedavg10'
				 'num-server-rounds'=10
				 'datasets'='[$dsets_cfg]'
				 " \
	--federation-config "
						options.num-supernodes=$((${#dsets[@]} - 1))
						options.backend.client-resources.num-cpus=1
						options.backend.client-resources.num-gpus=1
						" \
	2>&1 | tee ./logs/training-fedavg10.log

# run federated learning with server-side privacy
coreopsis run . standard \
	--stream \
	--run-config "
				 'fed-strategy'='FedAvg'
		         'output-home'='./output/fedavg10-ps'
		         'num-server-rounds'=10
				 'datasets'='[$dsets_cfg]'
				 'diff-priv-server'=1
				 'max-grad-norm'=1.0
                 'noise-multiplier'=1.5
				 " \
	--federation-config "
						options.num-supernodes=$((${#dsets[@]} - 1))
						options.backend.client-resources.num-cpus=1
						options.backend.client-resources.num-gpus=1
						" \
	2>&1 | tee ./logs/training-fedavg10-ps.log

# # try momentum
# coreopsis run . standard \
# 	--stream \
# 	--run-config "
# 				 'fed-strategy'='FedAvgM'
# 		         'output-home'='./output/fedavgm10'
# 		         'num-server-rounds'=10
# 				 'datasets'='[$dsets_cfg]'
# 				 "

# # try adam
# coreopsis run . standard \
# 	--stream \
# 	--run-config "
# 				 'fed-strategy'='FedAdam'
# 		         'output-home'='./output/fedadam10'
# 		         'num-server-rounds'=10
# 				 'datasets'='[$dsets_cfg]'
# 				 "

mdls=(
	"${dsets[@]/%/-p/mdl-cotorra}"
	# fedavg10/coreopsis-round-10
	# fedavg3/coreopsis-round-3
)

# extract reps for each dataset, for each model
for ds in "${dsets[@]}"; do
	for mdl in "${mdls[@]}"; do
		cotorra extract \
			--extraction-config ${config_home}/extraction.yaml \
			--processed-data-home ./processed/${ds} \
			--model-home ./output/${mdl} \
			--output-home "./processed/${ds}/mdl-$(dirname ${mdl})"
		cp ./processed/${ds}/*.{yaml,parquet} "./processed/${ds}/mdl-$(dirname ${mdl})"
		cotorra rep-based-score \
			--scoring-config ${config_home}/scoring.yaml \
			--processed-data-home "./processed/${ds}/mdl-$(dirname ${mdl})" \
			--model-home ./output/${mdl} \
			--estimator logistic-CV \
			--verbose \
			2>&1 | tee "./logs/scoring-ds-${ds}-mdl-$(dirname ${mdl}).log"
	done
done

python3 postprocessing.py
