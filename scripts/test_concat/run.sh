
eperc=(0 0.005 0.01 0.02 0.04 0.08)

for e in "${eperc[@]}"; do

  cd /home/nsccgz_qylin_1/IncrCP_paper/models/dlrm
  bash test_concat.sh $e

  cd /home/nsccgz_qylin_1/IncrCP_paper/scripts
  bash test_dlrm/test_concat_load.sh $e

done
# cd /home/nsccgz_qylin_1/IncrCP_paper/models/dlrm
# bash test_concat.sh 0.02

# cd /home/nsccgz_qylin_1/IncrCP_paper/scripts
# bash test_dlrm/test_concat_load.sh 0.02

# cd /home/nsccgz_qylin_1/IncrCP_paper/models/dlrm
# bash test_concat.sh 0.04

# cd /home/nsccgz_qylin_1/IncrCP_paper/scripts
# bash test_dlrm/test_concat_load.sh 0.04

cd /home/nsccgz_qylin_1/IncrCP_paper/models/dlrm
bash test_no_concat.sh

cd /home/nsccgz_qylin_1/IncrCP_paper/scripts
bash test_dlrm/test_no_concat_load.sh