import os, sys, datetime
import pandas as pd
import pickle
import nltk
import numpy as np
import tqdm
import os
from nltk.tokenize import *
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

def print_top_words(model, feature_names, n_top_words):
    tword = []
    for topic_idx, topic in enumerate(model.components_):
        print("Topic #%d:" % topic_idx)
        topic_w = " ".join([feature_names[i]+',' for i in topic.argsort()[:-n_top_words - 1:-1]])
        tword.append(topic_w)
        print(topic_w)
    return tword

file_path = './data'
os.chdir(file_path)

data = pd.read_csv("tokenized.csv",header = 0)
print(data)
n_features = 1000 #提取1000个特征词语
tf_vectorizer = CountVectorizer(strip_accents = 'unicode',
                                max_features=n_features,
                                stop_words='english',
                                max_df = 0.5,
                                min_df = 10,
                                ngram_range=(1, 2))
tf = tf_vectorizer.fit_transform(data['content'])

n_topics = 100
lda = LatentDirichletAllocation(n_components=n_topics, max_iter=50,
                                learning_method='batch',
                                learning_offset=50,
                                random_state=0)
lda.fit(tf)
print("TF ENDED")
n_top_words = 10
tf_feature_names = tf_vectorizer.get_feature_names()
topic_word = print_top_words(lda, tf_feature_names, n_top_words)
topics=lda.transform(tf)
print("LDA ENDDED")
P = 2
top_p_topics_indices = np.argsort(topics, axis=1)[:, ::-1][:, :P]
top_p_values = np.sort(topics, axis=1)[:, ::-1][:, :P]
data['assigned_topics'] = list(top_p_topics_indices)
data["topic_weight"] = list(top_p_values)
print(data[['content', 'assigned_topics','topic_weight']].head(5))

data.to_csv("data_topic.csv",index=False)
# with open('topic_list.pkl', 'wb') as f:
#     pickle.dump(topics, f)