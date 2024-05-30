import argparse
import time
import os

from incrcp import IncrCP
from naive_ckpt import NaiveCkpt

dlrm_emb_names = []
for i in range(0,26):
    dlrm_emb_names.append("emb_l." + str(i) + ".weight")

emb_names = []
for i in range(0, 39):
    emb_names.append("embedding." + str(i) + ".weight")

def recover_emb(args):
    if args.model_name == "dlrm":
        emb_names = dlrm_emb_names
    version = args.ckpt_version
    ckpt_dir = args.ckpt_dir + "/" + args.ckpt_method
    iter_num = version * args.ckpt_freq
    
    base_name = ckpt_dir + "/base." + str(iter_num) + ".pt"
    
    if os.path.exists(base_name) :
        emb_time = 0
        print(emb_time)
        return emb_time
    
    # emb_time_start = time.time()
    # load emb
    if args.ckpt_method == "incrcp":
        base_version = 0
        if version > args.incrcp_reset_thres :
            base_version = version // (args.incrcp_reset_thres + 1)
            version = version - (args.incrcp_reset_thres * base_version) - base_version
        
        method = IncrCP(ckpt_dir, emb_names, args.eperc, False, args.incrcp_reset_thres, base_version)
        # version in incrcp start from 0
        emb_time_start = time.time()
        emb = method.load_emb(version - 1)
        emb_time = time.time() - emb_time_start
    elif args.ckpt_method == "diff":
        method = NaiveCkpt(ckpt_dir, emb_names)
        emb_time_start = time.time()
        emb = method.load_emb(version,"diff")
        emb_time = time.time() - emb_time_start
    elif args.ckpt_method == "naive_incre":
        method = NaiveCkpt(args.ckpt_dir + "/" + args.ckpt_method, emb_names)
        emb_time_start = time.time()
        emb = method.load_emb(version, args.ckpt_method)
        emb_time = time.time() - emb_time_start
    # emb_time = time.time() - emb_time_start
    print(emb_time)
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
    args = parser.parse_args()
    
    
    
    emb_time = recover_emb(args)
    with open(args.perf_out_path, 'a') as log:
        log.write(str(emb_time) + "\n")
    
    # for i in range(1, args.ckpt_max_version + 1):
    #     t = recover_emb(args, i)
    #     perf["emb_recovery"].append(t)
    
    # with open(args.perf_out_path, "w") as json_file:
    #         json.dump(perf, json_file, indent=4)
    
    
        
if __name__ == "__main__":
    run()
