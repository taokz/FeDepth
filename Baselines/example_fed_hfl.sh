#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0


declare -a WIDTH=(0.165 0.33 0.5 1)
declare -a N_USER=(100)
declare -a N_CLASS=(-1 3)
declare -a R_USER=(10)
declare -a SLIM=(False True)
declare -a S_USER=(0 1 2)

#for n in "${N_USER[@]}"
for u in "${S_USER[@]}"
do
	for s in "${SLIM[@]}"
	do
		for c in "${N_CLASS[@]}"
		do
			file="fed_hfl.py"
			train="python $file --batch 128 --sel_user $u --slimmable_train $s --data Cifar10 --pd_nuser 100 --pu_nclass $c --pr_nuser 10 --wk_iters 5 --no_track_stat --lr 1e-2 --lr_sch cos"
			echo $train
			eval $train
			for w in "${WIDTH[@]}"
			do
					file="fed_hfl.py"
					test="python $file --batch 128 --sel_user $u --slimmable $s --data Cifar10 --pd_nuser 100 --pu_nclass $c --pr_nuser 10 --wk_iters 5 --no_track_stat --lr 1e-2 --lr_sch cos --test --test_slim_ratio $w"
					echo $test
					eval $test
			done
		done
	done
done

#for w in "${WIDTH[@]}"
#do
	#file="fed_splitmix.py"
	#test="python $file --data Cifar10 --pd_nuser $n --pu_nclass $c --pr_nuser 10 --wk_iters 5 --no_track_stat --lr 0.1 --lr_sch cos --test --test_slim_ratio $w"
	#echo $test
	#eval $test
#done

