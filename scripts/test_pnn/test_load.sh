
incrcp=$1
diff=$2
naive_incre=$3

ckpt_dir="/mnt/3dx/pnn"
reset=80
max_version=100
result_path=/home/nsccgz_qylin_1/IncrCP_paper/experimental_results/pnn

if [ $incrcp = 1 ]; then
  echo "incrcp"
  for ((i=0; i<=$max_version; i++)); do
    sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'
    python load_ckpt_deepfm_pnn.py --ckpt-method="incrcp" --ckpt-freq=10 --incrcp-reset-thres=$reset --ckpt-version=$i --ckpt-dir=$ckpt_dir \
     --perf-out-path="$result_path/incrcp.3dx.load"
  done
  # python -m line_profiler -rmt "load_ckpt.py.lprof" > incrcp.profile
fi

if [ $diff = 1 ]; then
  echo "diff"
  for ((i=0; i<=$max_version; i++)); do
    sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'
    python load_ckpt_deepfm_pnn.py --ckpt-method="diff" --ckpt-freq=10 --ckpt-version=$i --ckpt-dir=$ckpt_dir \
      --perf-out-path="$result_path/diff.3dx.load"
  done
  # python -m line_profiler -rmt "load_ckpt.py.lprof" > diff.profile
fi

if [ $naive_incre = 1 ]; then
  echo "naive_incre"
  for ((i=0; i<=$max_version; i++)); do
    sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'
    python load_ckpt_deepfm_pnn.py --ckpt-method="naive_incre" --ckpt-freq=10 --ckpt-version=$i --ckpt-dir=$ckpt_dir \
      --perf-out-path="$result_path/naive_incre.3dx.load"
  done
fi
