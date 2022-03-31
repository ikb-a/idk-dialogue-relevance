source env/bin/activate
export PYTHONPATH="."

python3 -u code/main.py \
--train_source lines/humod/humod_train_ctx.txt \
--test_source lines/humod/humod_test_ctx.txt \
--test_responses lines/humod/humod_test_res.txt \
--test_human lines/humod/humod_test_human.txt \
--output_dir lines/humod/test/

python3 -u code/main.py  \
--train_source lines/usr-tc-rel/tc_train_ctx.txt \
--test_source lines/usr-tc-rel/tc_a_test_ctx_3750.txt \
--test_responses lines/usr-tc-rel/tc_a_test_res_3750.txt \
--test_human lines/usr-tc-rel/tc_a_test_human_3750.txt \
--output_dir lines/usr-tc-rel/test_a/

python3 -u code/main.py  \
--train_source lines/pang_dd/pang_dd_ctx.txt \
--test_source lines/pang_dd/pang_dd_ctx.txt \
--test_responses lines/pang_dd/pang_dd_res.txt \
--test_human lines/pang_dd/pang_dd_human.txt \
--output_dir lines/pang_dd/test/

python3 -u code/main.py  \
--train_source lines/fed/fed_cor_ctx.txt \
--test_source lines/fed/fed_cor_ctx.txt \
--test_responses lines/fed/fed_cor_res.txt \
--test_human lines/fed/fed_cor_human.txt \
--output_dir lines/fed/test_fed_cor/

python3 -u code/main.py  \
--train_source lines/fed/fed_rel_ctx.txt \
--test_source lines/fed/fed_rel_ctx.txt \
--test_responses lines/fed/fed_rel_res.txt \
--test_human lines/fed/fed_rel_human.txt \
--output_dir lines/fed/test_fed_rel/
