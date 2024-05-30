import torch
import copy
import os


class Tracker:
    """
    Tracking updated embedding indexes
    """
    
    def __init__(self):
        self.diff_view = []
    
    # indexes corresponds to multiple layers
    def add(self, indexes):
        if len(self.diff_view) == 0:
            self.diff_view = copy.deepcopy(indexes)
        else :
            self.diff_view = [torch.unique(torch.cat((t1, t2))).sort().values 
                                    for t1, t2 in zip(self.diff_view, indexes)]
    
    def reset(self):
        self.diff_view = []

    def cur_view(self):
        return self.diff_view


def ckpt_non_emb(model, ckpt_path, cur_iter):
    to_save = {}
    for name, p in model.state_dict().items():
        if not "emb" in name:
            to_save[name] = p
    
    torch.save(to_save, ckpt_path + "/non-emb-" + str(cur_iter) + ".pt")


def get_folder_size(path):
    total_size = 0
    seen_inodes = set()

    for dirpath, dirnames, filenames in os.walk(path):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            if not os.path.exists(filepath):
                continue
            try:
                inode = os.stat(filepath).st_ino
                size = os.path.getsize(filepath)

                if inode in seen_inodes:
                    continue

                total_size += size
                seen_inodes.add(inode)

            except FileNotFoundError as e:
                print(f"Error accessing {filepath}: {e}")

    return total_size