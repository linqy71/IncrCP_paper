
incrcp=1
diff=1
naive_incre=1


ckpt_dir="/mnt/3dx/checkpoint" # the same as that in test_ckpt.sh
reset=100   # the same as --incrcp-reset-thres in test_ckpt.sh
max_version=150   # number of checkpoints created after running test_ckpt.sh
result_path=/home/nsccgz_qylin_1/IncrCP_paper/experimental_results/dlrm   # output path
ckpt_freq=10  # the same as that in test_ckpt.sh

if [ $incrcp = 1 ]; then
  echo "incrcp"
  for ((i=0; i<$max_version; i++)); do
    sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'
    python load_ckpt_dlrm.py --ckpt-method="incrcp" --ckpt-freq=$ckpt_freq --incrcp-reset-thres=$reset --ckpt-version=$i --ckpt-dir=$ckpt_dir \
     --perf-out-path="$result_path/dlrm.incrcp.log"
  done
fi

if [ $diff = 1 ]; then
  echo "diff"
  for ((i=0; i<$max_version; i++)); do
    sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'
    python load_ckpt_dlrm.py --ckpt-method="diff" --ckpt-freq=$ckpt_freq --ckpt-version=$i --ckpt-dir=$ckpt_dir \
      --perf-out-path="$result_path/dlrm.diff.log"
  done
fi

if [ $naive_incre = 1 ]; then
  echo "naive_incre"
  for ((i=0; i<$max_version; i++)); do
    sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'
    python load_ckpt_dlrm.py --ckpt-method="naive_incre" --ckpt-freq=$ckpt_freq --ckpt-version=$i --ckpt-dir=$ckpt_dir \
      --perf-out-path="$result_path/dlrm.naive_incre.log"
  done
fi


