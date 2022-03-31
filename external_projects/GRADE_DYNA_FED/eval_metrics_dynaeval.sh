# please specify your conda environment
source /home/administrator/anaconda3/etc/profile.d/conda.sh

BASE=`pwd`

# please specify the dataset you want to test here
DATA=(holistic)

export PATH=/usr/local/cuda-10.2/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-10.2/lib64:$LD_LIBRARY_PATH

cd ${BASE}/dynaeval
conda activate gcn

for data in ${DATA[@]}
do
    echo "Eval dynaeval $data"
    bash score.sh $data
done

