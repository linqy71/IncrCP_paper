import msgpack
import sys
import time


class NaiveCkpt:
    """
    Checkpointing embedding parameters using torch.save()
    """

    def __init__(
        self,
        db_root_path,
        emb_names,
    ):
        self.db_root_path = db_root_path
        self.emb_names = emb_names
        self.diff_hist = []
    
    def ckpt_emb(self, indexes, model, cur_iter):
        if len(indexes) != len(self.emb_names):
            sys.exit("ERROR: Given indexes do not match with model parameters.")
        
        # look up updated embedding vectors in batches
        to_save = {} # name: updated_param
        list_time = 0
        table_size = 0
        update_size = 0
        for i, name in enumerate(self.emb_names):
            updated_emb = {}
            emb_table = model.state_dict()[name]
            indices_tensor = indexes[i].clone().detach().to(emb_table.device)
            updated_emb_batch = emb_table[indices_tensor].detach().to('cpu')
            updated_emb_batch.numpy()
            
            indices_tensor = indices_tensor.to('cpu').numpy()
            lt = time.time()
            for k, idx in enumerate(indexes[i]):
                idx = int(idx)
                updated_emb[idx] = updated_emb_batch[k].tolist()
            lt = time.time() - lt
            list_time += lt
            to_save[name] = updated_emb
            update_size += len(updated_emb.keys())
            table_size += len(emb_table)
        
        save_time = time.time()
        file_name = self.db_root_path + "/data" + str(cur_iter) + ".cp"
        with open(file_name, 'wb') as file :
            data = msgpack.packb(to_save)
            file.write(data)
        save_time = time.time() - save_time
        return update_size / table_size, list_time, save_time
    
    def load_emb(self, version, method="diff", freq=10):
        if method == "diff":
            cur_iter = version * freq
            file_name = self.db_root_path + "/data" + str(cur_iter) + ".cp"
            with open(file_name, 'rb') as file:
                data = file.read()
                emb = msgpack.unpackb(data, strict_map_key=False)
        elif method == "naive_incre":
            result = {}
            for i in range(1, version + 1):
                cur_iter = i * freq
                file_name = self.db_root_path + "/data" + str(cur_iter) + ".cp"
                with open(file_name, 'rb') as file:
                    data = file.read()
                    emb = msgpack.unpackb(data, strict_map_key=False)
                if i == 1:
                    result = emb
                else :
                    for name, vecs in emb.items():
                        result[name].update(vecs)
            
    def may_reset_base(self, S):
        self.diff_hist.append(S)
        if sum(self.diff_hist) <= len(self.diff_hist) * S:
            return True
        else :
            return False

