#!/bin/bash

incrcp=1
diff=1
naive_incre=1


ckpt_dir="/GLOBALFS/nsccgz_zgchen_3/lqy/checkpoints/dis" # the same as that in test_ckpt.sh
reset=100   # the same as --incrcp-reset-thres in test_ckpt.sh
max_version=10   # number of checkpoints created after running test_ckpt.sh
result_path=/GLOBALFS/nsccgz_zgchen_3/lqy/IncrCP_paper/experimental_results/iter4000/rec_v5_10   # output path
ckpt_freq=4000  # the same as that in test_ckpt.sh

if [ $incrcp = 1 ]; then
  echo "incrcp"
  python load_ckpt_dist.py \
      --model-name="dlrm" \
      --ckpt-method="incrcp" \
      --ckpt-freq=$ckpt_freq \
      --ckpt-max-version=$max_version \
      --ckpt-dir=$ckpt_dir   \
      --perf-out-path="$result_path/incrcp.rec.json"
fi

if [ $diff = 1 ]; then
  echo "diff"
  python load_ckpt_dist.py \
      --model-name="dlrm" \
      --ckpt-method="diff" \
      --ckpt-freq=$ckpt_freq \
      --ckpt-max-version=$max_version \
      --ckpt-dir=$ckpt_dir   \
      --perf-out-path="$result_path/diff.rec.json"
fi

if [ $naive_incre = 1 ]; then
  echo "naive_incre"
  python load_ckpt_dist.py \
      --model-name="dlrm" \
      --ckpt-method="naive_incre" \
      --ckpt-freq=$ckpt_freq \
      --ckpt-max-version=$max_version \
      --ckpt-dir=$ckpt_dir   \
      --perf-out-path="$result_path/naive_incre.rec.json"
fi


