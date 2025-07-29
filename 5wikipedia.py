import time
import os
import pickle
import pandas as pd
from tqdm import tqdm
from wikipedia_sign import get_wikidata_description
from multiprocessing import Pool as ProcessPool
time0 = time.time()

def run(para, times = 0):
    ind = para[0]
    sentence = para[1]
    try:
        l = get_wikidata_description(sentence)
        if l is None:
            l='CANNOT FIND THIS ENTITY'
    except Exception as e:
        print(ind, e)
        print("Content: ", sentence)
        if times < 5:
            return run(para, times + 1)
        else:
            with open("error_info.txt", 'w+') as f:
                f.write("{}\t{}\n".format(ind, sentence))
            return None
    print(ind)
    return l

def process_pool(data):
    cnt = 0
    p = ProcessPool(8)
    chunkSize = 50000
    res = []
    i = 0
    while i < int(len(data)/chunkSize):
        try:
            res += list(p.map(run, data[i*chunkSize: (i+1)*chunkSize]))
            print(str(round((i+1)*chunkSize/len(data)*100, 2))+'%', round(time.time()-time0, 2))
            cnt += 1
            # fout = open("cache" + str(cnt).zfill(3) + '.txt', 'w', encoding='utf8')
            # fout.write('\n'.join(res))
            # fout.close()
            i += 1
            time.sleep(1.0)
        except:
            for i in range(60):
                time.sleep(10)
                print("\t{} / 600s".format(i*10))
        
    res += list(p.map(run, data[(i)*chunkSize:]))
    p.close()
    p.join()
    return res
# 使用示例
if __name__ == "__main__":
    file_path = './data'
    os.chdir(file_path)
    with open('./corpus.pkl', 'rb') as f:
        dict_list = pickle.load(f)

    desclist = process_pool(dict_list)

    with open('desc_list.pkl', 'wb') as f:
        pickle.dump(desclist, f)
