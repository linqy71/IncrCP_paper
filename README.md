# IncrCP: Decomposing and Orchestrating Incremental Checkpoints for Effective Recommendation Model Training

This package includes the source codes and testing scripts in the paper *IncrCP: Decomposing and Orchestrating Incremental Checkpoints for Effective Recommendation Model Training*.

```
.
├── src          
│   ├── ...
│   ├── two_d_chunk/            # Source code of 2DChunk
│   ├── incrcp.py               # Checkpointing interfaces of IncrCP
│   ├── naive_ckpt.py           # Checkpointing interfaces of Naive Incre and Check-N-Run
├── models
│   ├── deepfm                  
│   │   └── test_ckpt.sh        # Checkpoint construction test
│   │   └── ...
│   ├── dlrm                    
│   │   └── test_ckpt.sh        # Checkpoint construction test
│   │   └── ...
│   └── pnn                     
│   │   └── test_ckpt.sh        # Checkpoint construction test
│   │   └── ...
├── README.md
├── requirements.txt            # Python package requirements
└── scripts                     
    ├── load_ckpt.py
    ├── test_deepfm
    │   └── test_load.sh        # Recovery test
    ├── test_dlrm
    │   └── test_load.sh        # Recovery test
    └── test_pnn
        └── test_load.sh        # Recovery test

```

## Part 1: Preparations

1. clone this repo with:
```
git clone --recurse-submodules https://github.com/linqy71/IncrCP_paper.git
```

2. install package requirements
```
conda create --name incrcp python=3.8
pip install -r requirements.txt

git clone https://gitee.com/lgmcode/msgpack-c.git -b cpp_master
cd msgpack-c
mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX=/PATH/TO/msgpack-c ..
cmake --build . --target install
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/PATH/TO/msgpack-c/lib
export CPATH=$CPATH:/PATH/TO/msgpack-c/include

export CUDA_VISIBLE_DEVICES=0,1,2,3
```

3. install 2DChunk

```
cd src
python setup.py install
```

4. export the directory of interfaces to PYTHONPATH

```
export PYTHONPATH=/path/to/IncrCP_paper/src
```

5. Download [Criteo Kaggle Display Advertising Challenge Dataset](https://ailab.criteo.com/ressources/).


## Part 2: Result Reproduction

### Overall Performance reproduction

Here is our main overall performance results on different storage devices. 

![](./images/overall_perfromance_dlrm.jpg)

To reproduce this, first test `checkpoint construction` and then test `recovery`.

**1. test checkpoint construction**

```
cd models/dlrm
bash test_ckpt.sh
```
Before Runing scripts, make sure paths in the scripts is replaced to your own local paths.
Modify the following definitions to your need in `test_ckpt.sh`:
```
ckpt_dir=/path/to/save/checkpoints  # directory to save checkpoints
raw_data_file="/mnt/ssd/dataset/kaggle/train.txt"   # kaggle dataset path
processd_data="/mnt/ssd/dataset/kaggle/kaggleAdDisplayChallenge_processed.npz"  # kaggle dataset path
check_freq=10   # checkpoint frequency: number of iterations
num_batches=1500  # numebr of total training iterations
```

**2. test recovery**

```
cd scripts
bash test_dlrm/test_load.sh 
```
Before Runing scripts, make sure paths in the scripts is replaced to your own local paths.
Modify the following parameters in `test_load.sh`:
```
ckpt_dir="/mnt/3dx/checkpoint"  # the same as that in test_ckpt.sh
reset=100                      # to reset baseline ckpt of IncrCP
max_version=150               # number of checkpoints created after running test_ckpt.sh
result_path=/path/to/store/experimental_results/dlrm  # output path
ckpt_freq=10                  # the same as that in test_ckpt.sh
```