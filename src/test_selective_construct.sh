
eperc=(0 0.005 0.01 0.02 0.04 0.08 0.16 0.32 0.64 1)


for e in "${eperc[@]}"; do
rm -rf /home/nsccgz_qylin_1/checkpoint/incrcp
mkdir -p /home/nsccgz_qylin_1/checkpoint/incrcp

echo $e
python test_incrcp.py $e

done