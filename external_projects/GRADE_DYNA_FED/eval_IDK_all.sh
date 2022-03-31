# SCRIPT TO RUN ALL IDK experiments; note first need to run gen_IDK_all.sh to get data in right formats
#
# please specify your conda environment
source /home/administrator/anaconda3/etc/profile.d/conda.sh

BASE=`pwd`

# please specify the dataset you want to test here
DATA=(IDK_humod_test IDK_humod_val IDK_usr_tc_test IDK_usr_tc_val IDK_fed_cor IDK_fed_rel IDK_pang_dd_test)

export PATH=/usr/local/cuda-10.2/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-10.2/lib64:$LD_LIBRARY_PATH


# run grade
cd ${BASE}/grade
#conda activate grade_eval
source env_grade/bin/activate
#export PYTHONPATH=${BASE}/grade:PYTHONPATH
for data in ${DATA[@]}
do
    echo "Eval Grade $data"
    bash run_single.sh $data
done

# Run dynaeval
cd ${BASE}/dynaeval
conda activate gcn

for data in ${DATA[@]}
do
    echo "Eval dynaeval $data"
    bash score.sh $data
done
