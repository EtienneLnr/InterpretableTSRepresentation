#!/bin/bash

version=0

for reduc in 1 2 3
do
	Reduction=$((2 ** $reduc))
	Series_reduction=$(($2 / $Reduction))
	Seuil=2
	if [[ $Series_reduction -ge $Seuil ]]
	then
		python3 -m unsupervised_model.train_VQ_equi $1 --n_embed 16 --depth $reduc --version $version --num_training 10000 --lr 1e-4 --batch 128 --channel 64		
		python3 -m unsupervised_model.compute_ngrams_equi $1 --n_embed 16 --depth $reduc --version $version --channel 64
		version=$((version+1))
	else 
		continue
	fi 
	continue 
done

python3 -m classifiers.logistic_reg_equi_ensemble $1 --version $version