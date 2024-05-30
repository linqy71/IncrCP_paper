#!/bin/bash


paths=("/mnt/3dx/test_bw" "/mnt/ssd/test_bw" "/home/nsccgz_qylin_1/test_bw")


for path in "${paths[@]}"; do
    echo "Testing $path ..."

    mkdir -p $path

    echo "Sequential read/write test for $path"
    fio --name=seqwrite --rw=write --direct=1 --bs=1M --numjobs=1 --size=1G --runtime=60 --directory=$path
    fio --name=seqread --rw=read --direct=1 --bs=1M --numjobs=1 --size=1G --runtime=60 --directory=$path 

    echo "Random read/write test for $path"
    fio --name=randwrite --rw=randwrite --direct=1 --bs=4k --numjobs=1 --size=1G --runtime=60 --directory=$path 
    fio --name=randread --rw=randread --direct=1 --bs=4k --numjobs=1 --size=1G --runtime=60 --directory=$path

done

echo "All tests completed."
