import msgpack
import torch
import sys
import time
from concurrent.futures import ThreadPoolExecutor
import os
import copy

import py_tdchunk

def spilit_file_list(lst, parts):
    parts = max(1, parts)
    chunk_size = len(lst) // parts
    remainder = len(lst) % parts

    return [lst[i * chunk_size + min(i, remainder):(i + 1) * chunk_size + min(i + 1, remainder)]
            for i in range(parts)]

def read_and_deserialize(file_list):
    updated_emb = {}
    for file_name, beg, len in file_list:
        
        file = open(file_name, 'rb')
        file.seek(beg)
        data = file.read(len)
        emb = msgpack.unpackb(data, strict_map_key=False)

    return updated_emb

def ckpt_with_msgpack(db_root_path, name, cur_iter, updated_emb):
    cur_incre_data_name = db_root_path + '/' + name + "-data" + str(cur_iter) + ".msgpk"
    file = open(cur_incre_data_name, 'wb')
    data = msgpack.packb(updated_emb)
    file.write(data)
    file.close()
    return cur_incre_data_name

suffix = "tdc"
def construct(db_name, updated_emb, db_manager, i):
    file_number = db_manager.get_next_number(i)
    formatted_string = f"/{file_number:06}.{suffix}"
    cur_incre_data_name = db_name + '/' + formatted_string
    file = open(cur_incre_data_name, 'wb')
    data = msgpack.packb(updated_emb)
    length = len(data)
    file.write(data)
    file.close()
    keys = list(updated_emb.keys())
    db_manager.join(i, keys, file_number, length)

class IncrCP:
    """
    Checkpoint incremental data into LSEDB
    """
    
    def __init__(
        self,
        db_root_path,
        emb_names,
        ePerc=0.01,
        do_concat=False,
        reset_thres = 40,
        base_version = 0
    ):
        #open dbs
        self.db_manager = py_tdchunk.DBManager()
        self.db_names = []
        self.emb_names = emb_names
        self.do_concat = do_concat
        self.ePerc = ePerc
        self.base_version = base_version
        self.db_root_path = db_root_path
        cur_path = self.db_root_path + "/" + str(self.base_version)
        if not os.path.exists(cur_path):
            os.makedirs(cur_path, exist_ok=True)
        for name in emb_names:
            self.db_names.append(cur_path + '/' + name)
        # print(self.db_names)
        self.db_manager.open(self.db_names, do_concat, ePerc)
        self.reset_cnt = 0
        self.reset_thres = reset_thres
        # reset_thres: 50 for HDD, 100 for 3dx and flash SSD
    
    def finish(self):
        self.db_manager.releasedb()


    def ckpt_emb(self, diff_view, model, cur_iter):
        indexes = torch.from_numpy(diff_view.copy())
        
        if len(indexes) != len(self.emb_names):
            print(indexes)
            print(len(indexes))
            sys.exit("ERROR: Given indexes do not match with model parameters.")
        
        table_size = 0
        update_size = 0
        save_time = 0
        list_time = 0
        snapshot_time = 0
        for i, name in enumerate(self.emb_names):
            updated_emb = {}
            emb_table = model.state_dict()[name]
            snapshot = time.time()
            indices_tensor = indexes[i].detach().to(emb_table.device)
            updated_emb_batch = emb_table[indices_tensor].detach().to('cpu')
            updated_emb_batch.numpy()
            indices_tensor = indices_tensor.to('cpu').numpy()
            snapshot = time.time() - snapshot
            snapshot_time += snapshot
            
            lt = time.time()
            # updated_emb = {int(idx): emb_val.tolist() for idx, emb_val in zip(indices_tensor, updated_emb_batch)}
            for k, idx in enumerate(indexes[i]):
                idx = int(idx)
                updated_emb[idx] = updated_emb_batch[k].tolist()
            lt = time.time() - lt
            list_time += lt
            update_size += len(updated_emb.keys())
            table_size += len(emb_table)

            t = time.time()
            construct(self.db_names[i], updated_emb, self.db_manager, i)
            t = time.time() - t
            save_time += t

        return update_size / table_size, list_time, save_time, snapshot_time
    
    def load_emb(self, ckpt_version):
        result = {}
        total_len = 0
        for i, layer_name in enumerate(self.emb_names):
            file_list = py_tdchunk.getversion(self.db_manager, i, ckpt_version)
            
            updated_emb = {}
            for file_name, metas in file_list.items():
                with open(file_name, 'rb') as file:
                    for (start, length) in metas:
                        file.seek(start)
                        data = file.read(length)
                        total_len += length
                        emb = msgpack.unpackb(data, strict_map_key=False)
                        updated_emb.update(emb)
            result[layer_name] = updated_emb
        print(total_len / 1024 / 1024 / 1024)
        return result

    
    def load_emb_to_model(self, ckpt_version, model):
        """
        model: dict of parameters
        """
        for i, layer_name in enumerate(self.emb_names):
            file_list = py_tdchunk.getversion(self.db_manager, i, ckpt_version)
            
            updated_emb = {}
            cur_file = ""
            for file_name, beg, len in file_list:
                if cur_file != file_name:
                    cur_file = file_name
                    file = open(file_name, 'rb')
                # else no need to open file
                file.seek(beg)
                data = file.read(len)
                emb = msgpack.unpackb(data, strict_map_key=False)

                for idx, vec in emb.items():
                    updated_emb[idx] = torch.tensor(vec)
            
            # update embs to model
            if model:
                emb_table = model[layer_name]
                indices = torch.tensor(list(updated_emb.keys()))
                updates = torch.stack(list(updated_emb.values())).to(emb_table.device)
                emb_table.index_put_((indices,), updates)


    # def load_emb_multi_thread(self, ckpt_version):
        
    #     for i, layer_name in enumerate(self.emb_names):
    #         total_file_list = py_tdchunk.getversion(self.db_manager, i, ckpt_version)
    #         max_workers = 32
    #         file_lists = spilit_file_list(total_file_list, max_workers)
            
    #         updated_emb = {}
    #         with ThreadPoolExecutor(max_workers=max_workers) as executor:
    #             results = executor.map(read_and_deserialize, file_lists)
                

    def may_reset_base(self, S):
        self.reset_cnt += 1
        if self.reset_cnt >= self.reset_thres:
            self.reset_db()
            return True
        else :
            return False

    def reset_db(self):
        self.finish()
        self.base_version += 1
        cur_path = self.db_root_path + "/" + str(self.base_version)
        if not os.path.exists(cur_path):
            os.makedirs(cur_path)
        self.db_names = []
        for name in self.emb_names:
            self.db_names.append(cur_path + '/' + name)
        self.db_manager.open(self.db_names, self.do_concat, self.ePerc)
        self.reset_cnt = 0