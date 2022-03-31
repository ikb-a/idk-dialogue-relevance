# please specify your conda environment
source /home/administrator/anaconda3/etc/profile.d/conda.sh

BASE=`pwd`

# please specify the dataset you want to test here
DATA=(holistic)

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
