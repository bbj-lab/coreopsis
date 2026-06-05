#!/bin/bash

# tmux new -s co || tmux a -t co
# ln -s /mnt/bbj-lab/users/burkh4rt/data-raw ./data-raw

dsets=(
	"mimic-pre14"
	"mimic-post14"
	"ucmc-first"
)
config_home=./src/coreopsis/config

# collate data
for ds in "${dsets[@]}"; do
	cocoa collate \
		--collation-config ${config_home}/collation.yaml \
		--raw-data-home ./data-raw/${ds} \
		--processed-data-home ./processed/${ds}
done

# learn tokenizer on first dataset
cocoa tokenize \
	--tokenization-config ${config_home}/tokenization.yaml \
	--processed-data-home ./processed/${dsets[0]}

# apply tokenizer to other datasets
for ds in "${dsets[@]:1}"; do
	cocoa tokenize \
		--tokenization-config ${config_home}/tokenization.yaml \
		--tokenizer-home ./processed/${dsets[0]}/tokenizer.yaml \
		--processed-data-home ./processed/${ds}
done

# winnow data (prepare for inference)
for ds in "${dsets[@]}"; do
	cocoa winnow \
		--winnowing-config ${config_home}/winnowing.yaml \
		--processed-data-home ./processed/${ds}
done

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
		--output-home ./output/${ds}
done

# run federated learning
coreopsis run . standard \
	--stream \
	--run-config "
				 'fed-strategy'='FedAvg'
		         'output-home'='./output/fedavg10'
		         'num-server-rounds'=10
				 'datasets'='["mimic-pre14","mimic-post14","ucmc-first"]'
				 "

# try momentum
coreopsis run . standard \
	--stream \
	--run-config "
				 'fed-strategy'='FedAvgM'
		         'output-home'='./output/fedavgm10'
		         'num-server-rounds'=10
				 'datasets'='["mimic-pre14","mimic-post14","ucmc-first"]'
				 "

# extract reps for each dataset, for each model
for ds in "${dsets[@]}"; do
	for mdl in "fedavg10/coreopsis-round-10" \
		"mimic-pre14/mdl-coreopsis-training" \
		"mimic-post14/mdl-coreopsis-training" \
		"ucmc-first/mdl-coreopsis-training"; do
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
			--verbose
	done
done
