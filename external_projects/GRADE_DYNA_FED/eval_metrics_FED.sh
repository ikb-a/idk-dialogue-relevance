# please specify your conda environment
source /home/administrator/anaconda3/etc/profile.d/conda.sh

BASE=`pwd`

# please specify the dataset you want to test here
#DATA=(personachat_usr topicalchat_usr convai2_grade_bert_ranker convai2_grade_transformer_generator dailydialog_grade_transformer_ranker empatheticdialogues_grade_transformer_ranker convai2_grade_dialogGPT convai2_grade_transformer_ranker dailydialog_grade_transformer_generator  empatheticdialogues_grade_transformer_generator fed dstc6 dstc9 fed_dialog engage holistic)
DATA=(holistic)

# run usr_fed
cd ${BASE}
conda activate eval_base
for data in ${DATA[@]}
do
    echo "Eval USR FED $data"
    python usr_fed_metric.py --data $data
done
