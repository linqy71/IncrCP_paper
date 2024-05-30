

ckpt_dir=/mnt/ssd/checkpoint/concat

echo "load"
for i in {0..50}; do
  sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'
  python load_ckpt.py --ckpt-method="incrcp" --ckpt-freq=10 --ckpt-version=$i --ckpt-dir=$ckpt_dir --eperc=0 --perf-out-path="/home/nsccgz_qylin_1/IncrCP_paper/scripts/test_concat/outputs/no_concat.log"
done



