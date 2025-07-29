import os, sys, datetime
import pandas as pd
import pickle as pkl
import tqdm
import numpy as np
import contractions
import string
import os
from nltk.tokenize import *
data_name = "gossip.txt"
global NUM
NUM=0

def remove_punctuation(text):
    translator = str.maketrans('', '', string.punctuation)
    return text.translate(translator)

def word_cut(mytext):
    expanded_text = contractions.fix(mytext[0])
    striped = remove_punctuation(expanded_text)
    tokens = casual_tokenize(striped)
    filtered_tokens = [
        word.lower() for word in tokens 
        if word.lower() not in stopword_list
    ]
    return (" ").join(filtered_tokens)

file_path = './data'
stop_file = "./stopwords.txt"
os.chdir(file_path)
stopwords = open(stop_file,encoding ='utf-8')
stopword_list = stopwords.read().splitlines()
stopwords.close()

raw_data = pd.read_csv(data_name, header = 0)

tokenized = []
index_list = []
for index, line in raw_data.iterrows():
    sentences = sent_tokenize(line["content"])
    sent_num = []
    for sent in sentences:
        tokenized.append(sent)
        sent_num.append(NUM)
        NUM = NUM +1
    index_list.append(sent_num)

data = pd.DataFrame(tokenized)
data["content"] = data.apply(word_cut, axis=1)
data = data.drop([0], axis=1)

data.to_csv("tokenized.csv",index=False, header = True)

with open('sents.pkl', 'wb') as f:
     pkl.dump(index_list, f)