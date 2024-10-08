import msgpack
import sys
import time
import os
import torch
import copy


class NaiveCkpt:
    """
    Checkpointing embedding parameters using torch.save()
    """

    def __init__(
        self,
        db_root_path,
        emb_names
    ):
        self.db_root_path = db_root_path
        self.emb_names = emb_names
        self.file_root_path = []
        for name in emb_names:
            path = db_root_path + "/" + name
            if not os.path.exists(path):
                os.makedirs(path, exist_ok=True)
            self.file_root_path.append(path)
        self.diff_hist = []
    
    def ckpt_emb(self, diff_view, model, cur_iter):
        indexes = torch.from_numpy(diff_view.copy())
      
        if len(indexes) != len(self.emb_names):
            print(len(indexes), len(self.emb_names))
            sys.exit("ERROR: Given indexes do not match with model parameters.")
        
        # look up updated embedding vectors in batches
        list_time = 0
        save_time = 0
        snapshot_time = 0
        table_size = 0
        update_size = 0
        for i, name in enumerate(self.emb_names):
            updated_emb = {}
            emb_table = model.state_dict()[name]
            if len(indexes[i]) > (8000 * 1024): # too large, split
                index = indexes[i]
                num_parts = 10
                part_length = len(index) // num_parts
                parts = [index[i * part_length:(i + 1) * part_length] for i in range(num_parts)]
                for p in parts:
                    snapshot = time.time()
                    indices_tensor = p.detach().to(emb_table.device)
                    updated_emb_batch = emb_table[indices_tensor].detach().to('cpu')
                    updated_emb_batch.numpy()
                    indices_tensor = indices_tensor.to('cpu').numpy()
                    snapshot = time.time() - snapshot
                    snapshot_time += snapshot
                    lt = time.time()
                    for k, idx in enumerate(p):
                        idx = int(idx)
                        updated_emb[idx] = updated_emb_batch[k].tolist()
                    lt = time.time() - lt
                    list_time += lt
            else :
                snapshot = time.time()
                indices_tensor = indexes[i].detach().to(emb_table.device)
                updated_emb_batch = emb_table[indices_tensor].detach().to('cpu')
                updated_emb_batch.numpy()
                indices_tensor = indices_tensor.to('cpu').numpy()
                snapshot = time.time() - snapshot
                snapshot_time += snapshot
            
                lt = time.time()
                for k, idx in enumerate(indexes[i]):
                    idx = int(idx)
                    updated_emb[idx] = updated_emb_batch[k].tolist()
                lt = time.time() - lt
                list_time += lt
            update_size += len(updated_emb.keys())
            table_size += len(emb_table)
        
            t = time.time()
            file_name = self.file_root_path[i] + "/" + str(cur_iter) + ".cp"
            with open(file_name, 'wb') as file :
                data = msgpack.packb(updated_emb)
                file.write(data)
            t = time.time() - t
            save_time += t
        return update_size / table_size, list_time, save_time, snapshot_time
     
    
    def load_emb(self, version, method="diff", freq=10):
        result = {}
        if method == "diff":
            cur_iter = version * freq
            for i, root in enumerate(self.file_root_path):
                file_name = root + "/" + str(cur_iter) + ".cp"
                if not os.path.exists(file_name):
                    return result
                with open(file_name, 'rb') as file:
                    data = file.read()
                    emb = msgpack.unpackb(data, strict_map_key=False)
                result[self.emb_names[i]] = emb
        elif method == "naive_incre":
            for i in range(1, version + 1):
                cur_iter = i * freq
                for idx, root in enumerate(self.file_root_path):
                    file_name = root + "/" + str(cur_iter) + ".cp"
                    with open(file_name, 'rb') as file:
                        data = file.read()
                        vecs = msgpack.unpackb(data, strict_map_key=False)
                    if i == 1:
                        result[self.emb_names[idx]] = vecs
                    else :
                        result[self.emb_names[idx]].update(vecs)
        return result
            
    def may_reset_base(self, S):
        self.diff_hist.append(S)
        if sum(self.diff_hist) <= len(self.diff_hist) * S:
            return True
        else :
            return False

