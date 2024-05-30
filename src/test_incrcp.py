import py_tdchunk
import msgpack
import time
import sys

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

dlrm_emb_names = []
for i in range(0,26):
    dlrm_emb_names.append("emb_l." + str(i) + ".weight")
db_names = []
for i in range(0,26):
    db_names.append("/home/nsccgz_qylin_1/checkpoint/incrcp/" + dlrm_emb_names[i])

def main(argv):
    eperc = float(argv[1])
    db_manager = py_tdchunk.DBManager()
    db_manager.open(db_names, False, eperc)
    total_constrction_time = 0
    for iter in range(10, 500, 10):
        for i_db, db_name in enumerate(db_names):
            file = "/mnt/3dx/checkpoint/naive_incre/" + dlrm_emb_names[i_db] + "/" + str(iter) + ".cp"
            with open(file, "rb") as f:
              data = f.read()
            emb = msgpack.unpackb(data, strict_map_key=False)
            t = time.time()
            construct(db_name, emb, db_manager, i_db)
            t = time.time() - t
            total_constrction_time += t
    print(total_constrction_time)
    db_manager.releasedb()

if __name__ == "__main__":
    main(sys.argv)