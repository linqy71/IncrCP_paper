
e=$1

ckpt_dir=/home/nsccgz_qylin_1/checkpoint

echo "load"
for i in {0..50}; do
  sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'
  python load_ckpt.py --ckpt-method="incrcp" --ckpt-freq=10 --ckpt-version=$i --ckpt-dir=$ckpt_dir --eperc=$e --perf-out-path="./hdd_load_e$e.log"
done



