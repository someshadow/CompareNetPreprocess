import os, sys, datetime
import pandas as pd
import pickle as pkl
import torch
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from nltk.tokenize import *
from pykeen.pipeline import pipeline
from pykeen.datasets import PathDataset
from pykeen.triples import TriplesFactory

# nlp = spacy.load("en_core_web_sm")
# path1 = "./data/corpus.pkl"
# path2 = "./data/desc_list.pkl"

# with open(path1, 'rb') as f:
#     corpus_list = pkl.load(f)

# with open(path2, 'rb') as f:
#     desc_list = pkl.load(f)

# entity_desc_list = []
# # 从文档内容中收集词汇
# for entity, desc in zip(corpus_list,desc_list):
#     entity_desc_list.append([entity[1],'is',desc])

# data = pd.DataFrame(entity_desc_list)
# print(data)
# data.to_csv('./data/output.tsv', sep='\t', index=False)
os.chdir('./data/')
training_factory = TriplesFactory.from_path(
    path="output.tsv",
)
testing_factory = TriplesFactory.from_path(
    path="output.tsv",
)

result = pipeline(
    dataset=None,       # 内置数据集，也可用自定义文件路径
    training = training_factory,
    testing = testing_factory,
    model='TransE',              # 指定模型
    loss='marginranking',        # 损失函数
    training_loop='sLCWA',       # 训练模式
    epochs=100,                  # 训练轮数
    random_seed=42,
    device='cpu'
)

# 保存模型
# with open('entity_feature_transE.pkl', 'wb') as f:
#     pkl.dump(result, f)

# trained_model = torch.load('./transe_model.pkl')

result.save_to_directory('transe_model')

# 步骤7: 提取实体和关系嵌入矩阵
model = result.model
entity_embeddings = model.entity_representations[0]().detach().numpy()
relation_embeddings = model.relation_representations[0]().detach().numpy()

print("\n嵌入矩阵维度:")
print(f"实体嵌入: {entity_embeddings.shape}")  # (num_entities, embedding_dim)
print(f"关系嵌入: {relation_embeddings.shape}")  # (num_relations, embedding_dim)

# 步骤8: 创建实体到嵌入向量的映射
entity_id_to_label = training_factory.entity_id_to_label
entity_embedding_map = {
    entity_id_to_label[i]: entity_embeddings[i] 
    for i in range(len(entity_id_to_label))
}

# 步骤9: 保存嵌入矩阵
with open('entity_feature_transE.pkl', 'wb') as f:
     pkl.dump(entity_embeddings, f)
# np.save('entity_embeddings.npy', entity_embeddings)
# np.save('relation_embeddings.npy', relation_embeddings)

# print("\n示例实体嵌入:")
# for entity, embedding in list(entity_embedding_map.items())[:3]:
#     print(f"{entity[:30]}...: {embedding[:5]}...")