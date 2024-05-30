
incrcp=0
diff=0
naive_incre=1


ckpt_dir="/mnt/3dx/checkpoint"
reset=100
max_version=150
result_path=/home/nsccgz_qylin_1/IncrCP_paper/experimental_results/dlrm_100iter_ssd
ckpt_freq=10

if [ $incrcp = 1 ]; then
  echo "incrcp"
  for ((i=0; i<=$max_version; i++)); do
    sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'
    python load_ckpt_dlrm.py --ckpt-method="incrcp" --ckpt-freq=$ckpt_freq --incrcp-reset-thres=$reset --ckpt-version=$i --ckpt-dir=$ckpt_dir \
     --perf-out-path="$result_path/dlrm.incrcp.log"
  done
  # python -m line_profiler -rmt "load_ckpt.py.lprof" > incrcp.profile
fi

if [ $diff = 1 ]; then
  echo "diff"
  for ((i=0; i<=$max_version; i++)); do
    sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'
    python load_ckpt_dlrm.py --ckpt-method="diff" --ckpt-freq=$ckpt_freq --ckpt-version=$i --ckpt-dir=$ckpt_dir \
      --perf-out-path="$result_path/dlrm.diff.log"
  done
  # python -m line_profiler -rmt "load_ckpt.py.lprof" > diff.profile
fi

if [ $naive_incre = 1 ]; then
  echo "naive_incre"
  for ((i=150; i<=$max_version; i++)); do
    sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'
    python load_ckpt_dlrm.py --ckpt-method="naive_incre" --ckpt-freq=$ckpt_freq --ckpt-version=$i --ckpt-dir=$ckpt_dir \
      --perf-out-path="$result_path/dlrm.naive_incre.log"
  done
fi


