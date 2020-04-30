#!/bin/bash

source ~/opt/anaconda3/etc/profile.d/conda.sh
conda activate sgdml041e2

datasets=../Minis/H2Osplit/data_20_04_03/datasets/$1
models=../Minis/H2Osplit/data_20_04_03/models/$1

# c_model=''
# c_data=''

# for id in '1mer' '2body' '2mer' '3body' '3mer' '4body' '4mer' '5body' '5mer'
# do

# 	for data in $datasets/*${id}*.npz
# 	do 
# 		c_data=${data}
# 	done

# 	echo $c_data

# 	for model in $models/*${id}*.npz
# 	do
# 		c_model=${model}
# 	done

# 	echo $c_model

# 	python run.py cluster_error -i $c_model -d $c_data

# done

c_model=''
c_data=$datasets/${1}-4mer-dataset.npz

para_base=paras/mbGDML.py

cluster_file=clusters/John_${1}_c50.npy

for id in '1mer' '2body' '3body' '4body'
do

	# for model in $models/*${id}*.npz
	# do
	# 	c_model=${model}
	# done

	para_new=paras/mbGDML_${id}.py

	sed "s/'n_mers':4/'n_mers':${id::1}/g" $para_base > $para_new

	python run.py cluster_error -i $models -d $c_data -p $para_new -c $cluster_file

	# python run.py cluster_error -i $c_model -d $c_data

done