import argparse
import time
import os
import json

from incrcp import IncrCP
from naive_ckpt import NaiveCkpt


node0_slice = [
    [0,1],
    [0,1],
    [0,1],
    [0,1],
    [0,1],
    [0,1],
    [0,1],
    [0,1],
]

node1_slice = [
    [0,1],
    [0,1],
    [0],
    [0],
    [0],
    [0],
    [0],
    [0]
]


def recover_emb(args):
    version = args.ckpt_version 
    rank = args.local_rank
    node0_ckpt_dir = args.ckpt_dir +"/node0/" + args.ckpt_method
    node1_ckpt_dir = args.ckpt_dir +"/node1/" + args.ckpt_method
    
    if rank < 8 :
        ckpt_dir = node0_ckpt_dir
        my_slice = node0_slice
    else :
        ckpt_dir = node1_ckpt_dir
        my_slice = node1_slice
        rank = rank - 8
    
    ckpt_dir = ckpt_dir + "/rank" + str(rank)
    emb_names = []
    for i in my_slice[rank]:
        emb_names.append("emb_l." + str(i) + ".weight")
    
    method = NaiveCkpt(ckpt_dir, emb_names)
    emb_time_start = time.time()
    emb = method.load_emb(version, args.ckpt_method, args.ckpt_freq)
    emb_time = time.time() - emb_time_start
    
    return emb_time


def incrcp_recover_emb(args):
    version = args.ckpt_version 
    rank = args.local_rank
    node0_ckpt_dir = args.ckpt_dir +"/node0/" + args.ckpt_method
    node1_ckpt_dir = args.ckpt_dir +"/node1/" + args.ckpt_method
    
    if rank < 8 :
        ckpt_dir = node0_ckpt_dir
        my_slice = node0_slice
    else :
        ckpt_dir = node1_ckpt_dir
        my_slice = node1_slice
        rank = rank - 8
    
    ckpt_dir = ckpt_dir + "/rank" + str(rank)
    emb_names = []
    for i in my_slice[rank]:
        emb_names.append("emb_l." + str(i) + ".weight")
    
    method = IncrCP(ckpt_dir, emb_names, args.eperc, False, args.incrcp_reset_thres, base_version=0)
    emb_time_start = time.time()
    emb = method.load_emb(version - 1)
    emb_time = time.time() - emb_time_start
    
    return emb_time



def run():
    parser = argparse.ArgumentParser(
        description="Load Checkpoints"
    )
    parser.add_argument("--model-name", type=str, default="dlrm")
    
    #ckpt methods: diff, naive_incre, incrcp
    parser.add_argument("--ckpt-method", type=str, default="incrcp")
    #ckpt freq: every ckpt_freq iterations
    parser.add_argument("--ckpt-freq", type=int, default=10)
    parser.add_argument("--ckpt-version", type=int, default=10)
    parser.add_argument("--ckpt-dir", type=str, default="/mnt/ssd/checkpoint")
    parser.add_argument("--eperc", type=float, default=0.01)
    # pload: if true, load embs with multi threading, only when method==incrcp
    parser.add_argument("--pload", type=bool, default=False)
    parser.add_argument("--emb-only", type=bool, default=True)
    parser.add_argument("--perf-out-path", type=str, default="")
    parser.add_argument("--incrcp-reset-thres", type=int, default=100)
    
    parser.add_argument("--local-rank", type=int, default=0)
    args = parser.parse_args()
    
    
    if args.ckpt_method == "incrcp":
        t = incrcp_recover_emb(args)
    else :
        t = recover_emb(args)
    
    with open(args.perf_out_path, 'a') as file:
        file.write(str(t) + "\n")
    

if __name__ == "__main__":
    run()
