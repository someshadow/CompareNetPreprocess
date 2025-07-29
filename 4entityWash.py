import jsonline
import os
import pandas as pd
import json
import pickle
from tqdm import tqdm

def wash(data):
    dictlist = []
    for line in tqdm(data):
        temp = []
        for group in line:
            if group['rho'] > 0.3 and group['link_probability'] > 0.5:
                if 'title' in group:
                    temp.append(group['id'])
        dictlist.append(temp)
    return dictlist

def build_adj(data):
    dictlist = []
    for line in tqdm(data):
        temp = {}
        for group in line:
            if group['rho'] > 0.3 and group['link_probability'] > 0.5:
                if 'title' in group:
                    temp[group['id']] = {
                        'rho':group['rho'],
                        "link_probability":group['link_probability']
                        }
        dictlist.append(temp)
    return dictlist

def build_dict(data):
    dictlist = {}
    NUM = 0
    for line in tqdm(data):
        for group in line:
            if group['rho'] > 0.3 and group['link_probability'] > 0.5:
                if 'title' in group:
                    title = group['title']
                    group['id'] = NUM
                    NUM = NUM + 1
                    dictlist[title] = group
    return dictlist

if __name__ == "__main__":
    file_path = './data'
    os.chdir(file_path)
    to_entities = pd.read_csv(
        "tagme.txt",
        sep='\t',
        skip_blank_lines=True,
        engine='c',
        memory_map=True,
        header = None
    )
    send_list=[]
    for idx, line in tqdm(to_entities.iterrows()):
        try:
            if len(line) == 0:
                send_list.append([])
            send_list.append(json.loads(line[1]))
        except TypeError:
            send_list.append([])
    dictlist = build_dict(send_list)
    entitylist = wash(send_list)
    link_list = build_adj(send_list)
    tuplist = []
    NUM = 0
    for item in dictlist:
        tuplist.append((NUM,item))
        NUM = NUM+1

    with open('entity_list.pkl', 'wb') as f:
        pickle.dump(entitylist, f)
    with open('corpus.pkl', 'wb') as f:
        pickle.dump(tuplist, f)
    with open('link_diction.pkl', 'wb') as f:
        pickle.dump(link_list, f)


