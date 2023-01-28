#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=9

declare -a WIDTH=(0.165 0.33 0.5 1)
declare -a N_USER=(100)
declare -a P_DIR=(0.1 0.3 0.5)
declare -a R_USER=(20 10)
declare -a SEED=(1)
declare -a SLIM=(False True)

for s in "${SLIM[@]}"
do
        for r in "${R_USER[@]}"
        do
                for n in "${N_USER[@]}"
                do
                        for c in "${P_DIR[@]}"
                        do
                                file="fed_hfl.py"
                                train="python $file --slimmable_train $s --data Cifar10 --pd_nuser $n --pr_nuser $r --wk_iters 5 --no_track_stat --lr 1e-2 --lr_sch cos --additional DIR --partition_method dir --alpha $c"
                                echo $train
                                eval $train
                                for w in "${WIDTH[@]}"
                                do
                                        file="fed_hfl.py"
                                        test="python $file --slimmable_train $s --data Cifar10 --pd_nuser $n --pr_nuser $r --wk_iters 5 --no_track_stat --lr 1e-2 --lr_sch cos --additional DIR --partition_method dir --alpha $c --test --test_slim_ratio $w"
                                        echo $test
                                        eval $test
                                done
                        done
                done
        done
done