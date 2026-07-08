#!/bin/bash

# tmux new -s co || tmux a -t co

source .venv/bin/activate

dsets=(mimic-icu ucmc-icu nu-icu)
nsets=${#dsets[@]}
dsets_cfg=$(printf '"%s",' "${dsets[@]}")
dsets_cfg=${dsets_cfg%,}
config_home=./src/coreopsis/config
export dsets nsets dsets_cfg config_home

for h in mimic ucmc nu; do
	python recipes/run_clifpy.py \
		--data_dir "./data-raw/${h}-2.1.0" \
		--out_dir /scratch/$(whoami) \
		--waterfall \
		--convert_doses_continuous \
		--convert_doses_intermittent \
		2>&1 | tee ./logs/clifpy-${h}.log

	python recipes/run_sofa_scoring.py \
		--data_dir "./data-raw/${h}-2.1.0" \
		--out_dir /scratch/$(whoami) \
		2>&1 | tee ./logs/sofa-${h}.log
done

python3 recipes/preprocessing.py

# collate data
parallel --bar cocoa collate \
	--collation-config ${config_home}/collation.yaml \
	--raw-data-home ./data-raw/{} \
	--processed-data-home ./processed/{} \
	::: "${dsets[@]}"

# learn tokenizer on first dataset
cocoa tokenize \
	--tokenization-config ${config_home}/tokenization.yaml \
	--processed-data-home ./processed/${dsets[0]}

# apply tokenizer to other datasets
parallel --bar cocoa tokenize \
	--tokenization-config ${config_home}/tokenization.yaml \
	--tokenizer-home ./processed/${dsets[0]}/tokenizer.yaml \
	--processed-data-home ./processed/{} \
	::: "${dsets[@]:1}"

# winnow data (prepare for inference)
parallel --bar cocoa winnow \
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
	sbatch --export=ALL,ds=$ds,config_home=$config_home \
		recipes/run_training.sh
done

# train private models on each dataset
for ds in "${dsets[@]}"; do
	sbatch --export=ALL,private=1,ds=$ds,config_home=$config_home \
		recipes/run_training.sh
done

# run federated learning
sbatch --export=ALL \
	--gres=gpu:$nsets \
	recipes/run_federated.sh

# run federated learning with client-side privacy
sbatch --export=ALL,private=1 \
	--gres=gpu:$nsets \
	recipes/run_federated.sh

mdls=(
	${dsets[@]/%//mdl-cotorra}
	${dsets[@]/%/-p/mdl-cotorra}
	fedavg10/coreopsis-round-10
	fedavg10-p/coreopsis-round-10
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
			--estimator logistic-CV
	done
done

python3 recipes/postprocessing.py 2>&1 | tee ./logs/scoring.log
