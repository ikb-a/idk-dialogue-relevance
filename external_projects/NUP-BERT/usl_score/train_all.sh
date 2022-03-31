#!/bin/bash

source ../env3.6/bin/activate

export PYTHONPATH="."

echo HUMOD1

python3 -u train.py \
--metric NUP \
--weight-path exp_final/h/ \
--train-ctx-path lines/humod/humod_train_ctx.txt \
--train-res-path lines/humod/humod_train_res.txt \
--valid-ctx-path lines/humod/humod_valid_ctx.txt \
--valid-res-path lines/humod/humod_valid_res.txt \
--batch-size 16 \
--max-epochs 2 \
--ctx-token-len 100 \
--res-token-len 25

echo HUMOD2

python3 -u train.py \
--metric NUP \
--weight-path exp_final/h/ \
--train-ctx-path lines/humod/humod_train_ctx.txt \
--train-res-path lines/humod/humod_train_res.txt \
--valid-ctx-path lines/humod/humod_valid_ctx.txt \
--valid-res-path lines/humod/humod_valid_res.txt \
--batch-size 16 \
--max-epochs 2 \
--ctx-token-len 100 \
--res-token-len 25

echo HUMOD3

python3 -u train.py \
--metric NUP \
--weight-path exp_final/h/ \
--train-ctx-path lines/humod/humod_train_ctx.txt \
--train-res-path lines/humod/humod_train_res.txt \
--valid-ctx-path lines/humod/humod_valid_ctx.txt \
--valid-res-path lines/humod/humod_valid_res.txt \
--batch-size 16 \
--max-epochs 2 \
--ctx-token-len 100 \
--res-token-len 25

##########################################3


echo USR_TC_REL1

python3 -u train.py \
--metric NUP \
--weight-path exp_final/tc/ \
--train-ctx-path lines/usr-tc-rel/tc_train_ctx.txt \
--train-res-path lines/usr-tc-rel/tc_train_res.txt \
--valid-ctx-path lines/usr-tc-rel/tc_valid_ctx.txt \
--valid-res-path lines/usr-tc-rel/tc_valid_res.txt \
--batch-size 16 \
--max-epochs 2 \
--ctx-token-len 100 \
--res-token-len 25

echo USR_TC_REL2

python3 -u train.py \
--metric NUP \
--weight-path exp_final/tc/ \
--train-ctx-path lines/usr-tc-rel/tc_train_ctx.txt \
--train-res-path lines/usr-tc-rel/tc_train_res.txt \
--valid-ctx-path lines/usr-tc-rel/tc_valid_ctx.txt \
--valid-res-path lines/usr-tc-rel/tc_valid_res.txt \
--batch-size 16 \
--max-epochs 2 \
--ctx-token-len 100 \
--res-token-len 25

echo USR_TC_REL3

python3 -u train.py \
--metric NUP \
--weight-path exp_final/tc/ \
--train-ctx-path lines/usr-tc-rel/tc_train_ctx.txt \
--train-res-path lines/usr-tc-rel/tc_train_res.txt \
--valid-ctx-path lines/usr-tc-rel/tc_valid_ctx.txt \
--valid-res-path lines/usr-tc-rel/tc_valid_res.txt \
--batch-size 16 \
--max-epochs 2 \
--ctx-token-len 100 \
--res-token-len 25

#################################################


echo USR_TC_REL3-1

python3 -u train.py \
--metric NUP \
--weight-path exp_final/tc3750/ \
--train-ctx-path lines/usr-tc-rel/tc_train_ctx_3750.txt \
--train-res-path lines/usr-tc-rel/tc_train_res_3750.txt \
--valid-ctx-path lines/usr-tc-rel/tc_valid_ctx_3750.txt \
--valid-res-path lines/usr-tc-rel/tc_valid_res_3750.txt \
--batch-size 16 \
--max-epochs 2 \
--ctx-token-len 100 \
--res-token-len 25

echo USR_TC_REL3-2

python3 -u train.py \
--metric NUP \
--weight-path exp_final/tc3750/ \
--train-ctx-path lines/usr-tc-rel/tc_train_ctx_3750.txt \
--train-res-path lines/usr-tc-rel/tc_train_res_3750.txt \
--valid-ctx-path lines/usr-tc-rel/tc_valid_ctx_3750.txt \
--valid-res-path lines/usr-tc-rel/tc_valid_res_3750.txt \
--batch-size 16 \
--max-epochs 2 \
--ctx-token-len 100 \
--res-token-len 25

echo USR_TC_REL3-3

python3 -u train.py \
--metric NUP \
--weight-path exp_final/tc3750/ \
--train-ctx-path lines/usr-tc-rel/tc_train_ctx_3750.txt \
--train-res-path lines/usr-tc-rel/tc_train_res_3750.txt \
--valid-ctx-path lines/usr-tc-rel/tc_valid_ctx_3750.txt \
--valid-res-path lines/usr-tc-rel/tc_valid_res_3750.txt \
--batch-size 16 \
--max-epochs 2 \
--ctx-token-len 100 \
--res-token-len 25



