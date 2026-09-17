#!/bin/bash

# tmux new -s co || tmux a -t co

# "[Errno 11] Resource temporarily unavailable" typically indicates a mount problem
# try: sudo umount /mnt/bbj-lab && sudo mount /mnt/bbj-lab

source .venv/bin/activate

export vers=3.0.0
export config_home=./src/coreopsis/config

# harmonize respiratory data
for h in mimic ucmc nu rush eicu; do
	python recipes/run_clifpy.py \
		--data_dir "./data-raw/${h}-${vers}" \
		--out_dir /scratch/$(whoami) \
		--waterfall \
		2>&1 | tee ./logs/clifpy-${h}.log
done

python3 recipes/preprocessing.py

export dsets=(
	mimic-icu-3.0.0
	ucmc-icu-3.0.0
	nu_cdh-icu-3.0.0
	nu_dh-icu-3.0.0
	nu_hh-icu-3.0.0
	nu_kh-icu-3.0.0
	nu_lfh-icu-3.0.0
	nu_mh-icu-3.0.0
	nu_nmh-icu-3.0.0
	nu_ph-icu-3.0.0
	rush-icu-3.0.0
	eicu-73-icu-3.0.0
	eicu-122-icu-3.0.0
	eicu-148-icu-3.0.0
	eicu-165-icu-3.0.0
	eicu-167-icu-3.0.0
	eicu-176-icu-3.0.0
	eicu-183-icu-3.0.0
	eicu-188-icu-3.0.0
	eicu-199-icu-3.0.0
	eicu-208-icu-3.0.0
	eicu-243-icu-3.0.0
	eicu-248-icu-3.0.0
	eicu-252-icu-3.0.0
	eicu-264-icu-3.0.0
	eicu-281-icu-3.0.0
	eicu-283-icu-3.0.0
	eicu-300-icu-3.0.0
	eicu-307-icu-3.0.0
	eicu-331-icu-3.0.0
	eicu-338-icu-3.0.0
	eicu-345-icu-3.0.0
	eicu-365-icu-3.0.0
	eicu-394-icu-3.0.0
	eicu-411-icu-3.0.0
	eicu-413-icu-3.0.0
	eicu-416-icu-3.0.0
	eicu-417-icu-3.0.0
	eicu-420-icu-3.0.0
	eicu-443-icu-3.0.0
	eicu-449-icu-3.0.0
	eicu-458-icu-3.0.0
)
export dsets_csv=$(printf '%s,' "${dsets[@]}")

# collate data
parallel --bar cocoa collate \
	--collation-config ${config_home}/collation.yaml \
	--raw-data-home ./data-raw/{} \
	--processed-data-home ./processed/{} \
	--verbose \
	::: "${dsets[@]}" \
	2>&1 | tee ./logs/collation.log

# learn tokenizer on first dataset
cocoa tokenize \
	--tokenization-config ${config_home}/tokenization.yaml \
	--processed-data-home ./processed/${dsets[0]} \
	2>&1 | tee ./logs/tokenization-0.yaml

# apply tokenizer to other datasets
parallel --bar cocoa tokenize \
	--tokenization-config ${config_home}/tokenization.yaml \
	--tokenizer-home ./processed/${dsets[0]}/tokenizer.yaml \
	--processed-data-home ./processed/{} \
	::: "${dsets[@]:1}" \
	2>&1 | tee ./logs/tokenization-1+.yaml

# winnow data (prepare for inference)
parallel --bar cocoa winnow \
	--winnowing-config ${config_home}/winnowing.yaml \
	--processed-data-home ./processed/{} \
	::: "${dsets[@]}" \
	2>&1 | tee ./logs/winnowing.yaml

# create a combined dataset
cocoa combine-datasets \
	"${dsets[@]/#/./processed/}" \
	--output-data-dir ./processed/all

dsets+=('all')

# # train separate models on each dataset
# for ds in "${dsets[@]}"; do
# 	sbatch --export=ALL,ds=$ds,config_home=$config_home \
# 		recipes/run_training.sh
# done

# # pull out and rename models saved at each 1/100th part of the data
# for c in c-{ucmc,nu,mimic}-icu; do
# 	i=0
# 	for d in $(ls -dtr ./output/$c/checkpoint-*); do
# 		printf -v new "./output/$c-%03d" "$((++i))"
# 		mkdir -p "$new/mdl-cotorra" && cp -a "$d/." "$new/mdl-cotorra"
# 	done
# 	mkdir -p ./output/$c-100/mdl-cotorra
# 	cp -a ./output/$c/mdl-cotorra/. ./output/$c-100/mdl-cotorra
# done

# # GEM-* runs
# for ds in "${dsets[@]}"; do
# 	sbatch --export=ALL,ds=$ds,config_home=$config_home \
# 		recipes/run_star_training.sh
# done

# # pull out and rename models saved at each 1/5th part of the 5 epoch run
# for c in cxxx-{ucmc-icu,nu-icu,mimic-icu,all}; do
# 	i=0
# 	for d in $(ls -dtr ./output/$c/checkpoint-*); do
# 		printf -v new "./output/$c-%03d" "$((++i))"
# 		mkdir -p "$new/mdl-cotorra" && cp -a "$d/." "$new/mdl-cotorra"
# 	done
# done

# # ablate over server rounds
# for num_server_rounds in 1 5 10 50; do
# 	export num_server_rounds
# 	dsets=(mimic-icu ucmc-icu nu-icu)
# 	nsets=${#dsets[@]}
# 	dsets_cfg=$(printf '"%s",' "${dsets[@]}")
# 	dsets_cfg=${dsets_cfg%,}
# 	output_home="./output/c-fedavg${num_server_rounds}"
# 	export dsets nsets dsets_cfg output_home
# 	sbatch --export=ALL \
# 		--gres=gpu:$nsets \
# 		recipes/run_federated.sh
# done

# # ablate over strategy
# for fed_strategy in FedAvgM FedAdam; do
# 	export fed_strategy
# 	export num_server_rounds=10

# 	# run federated learning on all datasets
# 	dsets=(mimic-icu ucmc-icu nu-icu)
# 	nsets=${#dsets[@]}
# 	dsets_cfg=$(printf '"%s",' "${dsets[@]}")
# 	dsets_cfg=${dsets_cfg%,}
# 	output_home="./output/c-${fed_strategy,,}${num_server_rounds}"
# 	export dsets nsets dsets_cfg output_home
# 	sbatch --export=ALL \
# 		--gres=gpu:$nsets \
# 		recipes/run_federated.sh
# done

# # rep-based scoring
# for ds in mimic-icu ucmc-icu nu-icu; do
# 	mdls=(
# 		cxxx-{mimic-icu,ucmc-icu,nu-icu,all}-005/mdl-cotorra
# 		c-${ds}-{{001..010},{015..100..5}}/mdl-cotorra
# 		c-fedavg1/coreopsis-round-1
# 		c-fedavg5/coreopsis-round-5
# 		c-fedavg10{,-mc,-mn,-cn}/coreopsis-round-10
# 		c-fed{avgm,adam}10/coreopsis-round-10
# 		c-fedavg50/coreopsis-round-50
# 		c-all/mdl-cotorra
# 	)
# 	for mdl in "${mdls[@]}"; do
# 		cotorra extract \
# 			--extraction-config ${config_home}/extraction.yaml \
# 			--processed-data-home ./processed/${ds} \
# 			--model-home ./output/${mdl} \
# 			--output-home "./processed/${ds}/mdl-$(dirname ${mdl})"
# 		cp ./processed/${ds}/*.{yaml,parquet} "./processed/${ds}/mdl-$(dirname ${mdl})"
# 		cotorra rep-based-score \
# 			--scoring-config ${config_home}/scoring.yaml \
# 			--processed-data-home "./processed/${ds}/mdl-$(dirname ${mdl})" \
# 			--model-home ./output/${mdl} \
# 			--estimator logistic-CV
# 	done
# done

# python3 recipes/baselines.py
# python3 recipes/postprocessing.py 2>&1 | tee ./logs/postprocessing.log
# python3 recipes/tokenwise.py
# python3 recipes/plotting.py
# python3 recipes/analyze-site-data.py
