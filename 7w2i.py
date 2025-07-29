import os, sys, datetime
import pandas as pd
import pickle as pkl
import tqdm
import numpy as np
import contractions
import string
import os
from nltk.tokenize import *

path1 = "./data/desc_list.pkl"
path2 = "./data/tokenized.csv"

with open(path1, 'rb') as f:
    desc_list = pkl.load(f)

raw_data = pd.read_csv(path2, header = 0)
all_docs_content = []
for index, row in raw_data.iterrows():
    sent = row['content']
    all_docs_content.append(sent)

vocab = set(['<pad>', '<unk>'])
# 从文档内容中收集词汇
for doc in all_docs_content:
    try:
        token = word_tokenize(doc)
    except TypeError:
        continue
    for word in token:
        vocab.add(word)
        

# 从实体描述中收集词汇
for desc in desc_list:
    for sent in sent_tokenize(desc):
        for word in word_tokenize(sent.lower()):
            vocab.add(word)
        
w2i = {word: i for i, word in enumerate(vocab)}

with open('./data/w2i.pkl', 'wb') as f:
    pkl.dump(w2i, f)
