#!/usr/bin/env python
# -*- coding: utf-8 -*-
import time, datetime
import json
import requests
import os
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool as ProcessPool
time0 = time.time()

def getEntityList(text):
    url = 'https://tagme.d4science.org/tagme/tag'
    # token = '14a339a1-6913-4e8a-bff3-da6ae4459381-843339462'
    token = 'fe4df7bf-ab75-4efb-aa1c-551afaa65cd3-843339462'
    # token ='794d0715-1591-4919-bbbb-84b00a665335-843339462'
    headers = {
        'Host': 'tagme.d4science.org',
        'Connection': 'keep-alive',
        'Cache-Control': 'max-age=0',
        'Upgrade-Insecure-Requests': '1',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/63.0.3239.132 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
        'Accept-Encoding': 'gzip, deflate, br',
        'Accept-Language': 'zh-CN,zh;q=0.9',
        'Cookie': '_ga=GA1.2.827175290.1544765315; _gid=GA1.2.121830695.1544765315',
    }
    send = {
        "gcube-token": token,
        "text": text,
        "lang": "en"
    }
    response = requests.post(url,data=send, headers=headers, timeout=100)
    try:
        return json.loads(response.text)['annotations']
    except:
        print('.'*50)
        print(text)
        print(response.text)
        print('.'*50)

def sentence2Link(sentence):
    return json.dumps(getEntityList(sentence))


def run(para, times = 0):
    ind = para[0]
    sentence = para[1]
    try:
        l = sentence2Link(sentence)
        if l is None:
            l = 'ERROR'
    except Exception as e:
        print(ind, e)
        print("Content: ", sentence)
        if times < 5:
            return run(para, times + 1)
        else:
            with open("error_info.txt", 'w+') as f:
                f.write("{}\t{}\n".format(ind, sentence))
            return "ERROR"
    print(ind)
    return str(ind)+'\t'+l


def process_pool(data):
    cnt = 0
    p = ProcessPool(32)
    chunkSize = 128
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

if __name__ == "__main__":
    file_path = './data'
    os.chdir(file_path)
    raw_data=pd.read_csv("tokenized.csv")
    data = []
    for index, row in raw_data.iterrows():
        send_list = [index,row['content']]
        data.append(send_list)

    print('read done. Tagging...')
    print(len(data))
    outdata = process_pool(data)
    fout = open("./tagme.txt", 'w', encoding='utf8')
    fout.write('\n'.join(outdata))
    fout.close()