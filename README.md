# LSECP:A high-performance and space-efficient checkpoint system based on LSE-Tree for recommendation models

This package includes the source codes and testing scripts in the paper *LSECP:A high-performance and space-efficient checkpoint system based on LSE-Tree for recommendation models*.

```
.
├── src          
│   ├── ...
│   ├── two_d_chunk/            # Source code of 2DChunk
│   ├── incrcp.py               # Checkpointing interfaces of IncrCP
│   ├── naive_ckpt.py           # Checkpointing interfaces of Naive Incre and Check-N-Run
├── models
│   ├── deepfm                  
│   │   └── test_ckpt.sh        # script for testing checkpointing
│   │   └── ...
│   ├── dlrm                    
│   │   └── test_ckpt.sh        # script for testing checkpointing
│   │   └── ...
│   └── pnn                     
│   │   └── test_ckpt.sh        # script for testing checkpointing
│   │   └── ...
├── README.md
├── requirements.txt            # Python package requirements
└── scripts                     
    ├── load_ckpt.py
    ├── test_deepfm
    │   └── test_load.sh        # script for testing recovery
    ├── test_dlrm
    │   └── test_load.sh        # script for testing recovery
    └── test_pnn
        └── test_load.sh        # script for testing recovery

```

## Part 1: Getting Started Guide

clone this repo with:
```
git clone --recurse-submodules 
```

### Environment Setups

1. install package requirements
```
conda create --name incrcp python=3.8
pip install -r requirements.txt

export CUDA_VISIBLE_DEVICES=0,1,2,3
```

2. install 2DChunk

```
cd src
python setup.py install
```

export the directory of interfaces to PYTHONPATH
```export PYTHONPATH=/path/to/IncrCP_paper/src```

## Part 2: Run checkpoint tests

