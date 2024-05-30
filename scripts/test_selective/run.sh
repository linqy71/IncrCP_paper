# eperc=(0 0.005 0.01 0.02 0.04 0.08 0.16 0.32 0.64 1)
# eperc=(0.08 0.16 0.32 0.64 1)

eperc=(0.005 0.01 0.02 0.04 0.08 0.16 0.32 0.64 1 0)
# eperc=(0)

for e in "${eperc[@]}"; do
  cd /home/nsccgz_qylin_1/IncrCP_paper/models/dlrm
  bash test_incrcp_eperc.sh $e

  # cd /home/nsccgz_qylin_1/IncrCP_paper/scripts
  # bash test_dlrm/test_selective_load.sh $e
done