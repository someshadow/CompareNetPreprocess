import os, sys, datetime
import pandas as pd
import re
import nltk
import networkx as nx
import numpy as np
import torch
import tqdm
import pickle as pkl
from nltk.tokenize import *
from str2float import *
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

def build_something_to_input(sent_index_list, entity_index_list, topic_index_list):
    indexdict = {}
    n = 0
    for idx in range(0,len(sent_index_list)):
        num = 0
        sent_index_dict = {}
        entity_index_dict = {}
        topic_index_dict = {}
        temp = []
        for sent in sent_index_list[idx]:
            sent_index_dict[num] = sent
            for t in topic_index_list[sent]:
                temp.append(t)
            num = num+1
        temp_list = []
        for item in temp:
            temp_list.append(item[0])
        temp = set(temp_list)
        num = 0
        for topic in temp:
            topic_index_dict[num] = topic
            num = num+1
        num = 0
        for ent in entity_index_list[idx]:
            entity_index_dict[num] = ent
            num = num+1
        indexdict[n]={
            'idx':idx,
            'adj_list':[],
            's2i':sent_index_dict,
            'e2i':entity_index_dict,
            't2i':topic_index_dict,
            'sentNum':len(sent_index_dict)
        }
        n=n+1
    return indexdict
        

def build_heterogeneous_graph(data, sentence_entity_weights, sentence_topic_weights):
    """
    构建新闻异构图
    
    参数:
        s2i: 句子节点映射 {图节点ID: 外部句子库ID}
        e2i: 实体节点映射 {图节点ID: 外部实体库ID}
        t2i: 主题节点映射 {图节点ID: 外部主题库ID}
        sentence_entity_edges: 句子-实体关系 [(句子节点ID, 实体节点ID)]
        sentence_topic_edges: 句子-主题关系 [(句子节点ID, 主题节点ID)]
    """
    s2i = data['s2i']
    e2i = data['e2i']
    t2i = data['t2i']
    # sentence_entity_edges = link_sentences_entity(s2i, e2i, sentence_entity_edges)
    # sentence_topic_edges = link_sentences_topic(s2i, t2i, sentence_topic_edges)
    sentence_entity_edge_and_weight = link_prob_entity(s2i, e2i, sentence_entity_weights)
    sentence_topic_edge_and_weight = link_prob_topic(s2i, t2i, sentence_topic_weights)
    # 创建有向图
    G = nx.DiGraph()
    
    # 添加节点（区分类型）
    # 句子节点
    for node_id, external_id in s2i.items():
        G.add_node(('s', node_id), node_type='s', external_id=external_id, label=f"S{node_id}")
    
    # 实体节点
    for node_id, external_id in e2i.items():
        G.add_node(('e', node_id), node_type='e', external_id=external_id, label=f"E{node_id}")
    
    # 主题节点
    for node_id, external_id in t2i.items():
        G.add_node(('t', node_id), node_type='t', external_id=external_id, label=f"T{node_id}")
    
    # 1. 句子-句子全连接（包括自环）
    n_s = len(s2i)
    for s1 in s2i.keys():
        for s2 in s2i.keys():
            weight = 1.0 / (n_s + 2)  # 归一化权重
            G.add_edge(('s', s1), ('s', s2), weight=weight, relation='s-s')
    
    # 2. 实体-实体自环
    for e in e2i.keys():
        G.add_edge(('e', e), ('e', e), weight=0.5, relation='e-e')
    
    # 3. 主题-主题自环
    for t in t2i.keys():
        G.add_edge(('t', t), ('t', t), weight=0.5, relation='t-t')
    
    # 4. 句子-实体双向连接
    for item in sentence_entity_edge_and_weight:
        sent_id = item[0]
        ent_id = item[1]
        weight_num = item[2]
        # 句子 → 实体
        G.add_edge(('s', sent_id), ('e', ent_id), weight=weight_num, relation='s-e')
        # 实体 → 句子
        G.add_edge(('e', ent_id), ('s', sent_id), weight=weight_num, relation='e-s')
    
    # 5. 句子-主题双向连接
    for item in sentence_topic_edge_and_weight:
        sent_id = item[0]
        topic_id = item[1]
        weight_num = item[2]
        # 句子 → 主题
        G.add_edge(('s', sent_id), ('t', topic_id), weight=weight_num, relation='s-t')
        # 主题 → 句子
        G.add_edge(('t', topic_id), ('s', sent_id), weight=weight_num, relation='t-s')

    # === 新增：按关系类型分别归一化权重 ===
    # 获取所有关系类型
    relations = set(data['relation'] for _, _, data in G.edges(data=True))
    
    # 按关系类型分组线性归一化
    for rel in relations:
        # 获取该关系类型的所有边及其权重
        edges = [(u, v) for u, v, data in G.edges(data=True) if data['relation'] == rel]
        weights = [G[u][v]['weight'] for u, v in edges]
        
        min_weight = min(weights)
        max_weight = max(weights)
        range_weight = max_weight - min_weight

        if range_weight > 1e-10:
            for u, v in edges:
                normalized = (G[u][v]['weight'] - min_weight) / range_weight
                G[u][v]['weight'] = normalized

    return G

def link_sentences_entity(sentence2i, entity2i, refer_list):
    dict_list = []
    for sentence_item in sentence2i.items():
        for entity_item in entity2i.items():
            for entity_id in refer_list[sentence_item[1]]:
                if entity_id == entity_item[1]:
                    dict_list.append([sentence_item[0],entity_item[0]])
    return dict_list

def link_sentences_topic(sentence2i, topic2i, refer_list):
    dict_list = []
    for sentence_item in sentence2i.items():
        for topic_item in topic2i.items():
            for entity_id in refer_list[sentence_item[1]]:
                if entity_id == topic_item[1]:
                    dict_list.append([sentence_item[0],topic_item[0]])
    return dict_list
    
def link_prob_entity(sentence2i, entity2i, refer_list):
    dict_list = []
    for sentence_item in sentence2i.items():
        for entity_item in entity2i.items():
            for entity_id in refer_list[sentence_item[1]].items():
                if entity_id[0] == entity_item[1]:
                    dict_list.append([sentence_item[0],entity_item[0],entity_id[1]['link_probability']])
    return dict_list

def link_prob_topic(sentence2i, topic2i, refer_list):
    dict_list = []
    for sentence_item in sentence2i.items():
        for topic_item in topic2i.items():
            for topic_id in refer_list[sentence_item[1]]:
                if topic_id[0] == topic_item[1]:
                    dict_list.append([sentence_item[0],topic_item[0],topic_id[1]])
    return dict_list

def extract_adjacency_matrices(G):
    """
    从图中提取9种关系的邻接矩阵
    
    返回:
        adj_list: 包含9个邻接矩阵的列表
    """
    # 初始化邻接矩阵列表 (9种关系)
    adj_list = []
    
    # 定义9种关系类型
    relations = [
        ('s', 's'),  # 0: s→s
        ('s', 'e'),  # 1: s→e
        ('s', 't'),  # 2: s→t
        ('e', 's'),  # 3: e→s
        ('e', 'e'),  # 4: e→e
        ('e', 't'),  # 5: e→t
        ('t', 's'),  # 6: t→s
        ('t', 'e'),  # 7: t→e
        ('t', 't')   # 8: t→t
    ]
    
    # 为每种关系创建邻接矩阵
    for src_type, dst_type in relations:
        # 收集所有符合条件的边
        edges = []
        for u, v, data in G.edges(data=True):
            # 检查源节点类型和目标节点类型
            if G.nodes[u]['node_type'] == src_type and G.nodes[v]['node_type'] == dst_type:
                # 获取节点索引（去掉类型前缀）
                src_idx = u[1]
                dst_idx = v[1]
                weight = data['weight']
                edges.append((src_idx, dst_idx, weight))
        
        # 如果没有边，创建空矩阵
        if not edges:
            # 确定矩阵大小
            n_src = len([n for n, d in G.nodes(data=True) if d['node_type'] == src_type])
            n_dst = len([n for n, d in G.nodes(data=True) if d['node_type'] == dst_type])
            
            adj_list.append({
                'indices': torch.empty((2, 0), dtype=torch.long),
                'value': torch.tensor([]),
                'size': torch.Size([n_src, n_dst])
            })
            continue
        
        # 提取源节点索引、目标节点索引和权重
        src_indices, dst_indices, weights = zip(*edges)
        
        # 创建COO格式的邻接矩阵
        indices = torch.tensor([src_indices, dst_indices], dtype=torch.long)
        values = torch.tensor(weights, dtype=torch.float32)
        
        # 确定矩阵大小
        n_src = len(set(src_indices))
        n_dst = len(set(dst_indices))
        
        adj_list.append({
            'indices': indices,
            'value': values,
            'size': torch.Size([n_src, n_dst])
        })
    
    return adj_list
    
if __name__ == "__main__":
    path = {
        "sent_index_list":"./data/sents.pkl",
        "entity_index_list":"./data/entity_list.pkl",
        "topic_index_list":"./data/data_topic.csv",
        "entity_occurrences":"./data/entity_list.pkl",
        "entity_weight":"./data/link_diction.pkl"
    }
    with open(path['sent_index_list'], 'rb') as f:
        sent_index_list = pkl.load(f)
    with open(path['entity_index_list'], 'rb') as f:
        entity_index_list = pkl.load(f)
    topic_temp = pd.read_csv(path['topic_index_list'],header = 0).drop(['content'],axis = 1)
    topic_index_list = topic_temp.apply(convert_row_robust, axis=1).tolist()
    topic_index_list = transform_list(topic_index_list)
    inputdict = build_something_to_input(sent_index_list, entity_index_list, topic_index_list)
    with open(path['entity_occurrences'], 'rb') as f:
        entity_occurrences = pkl.load(f)
    with open(path['entity_weight'], 'rb') as f:
        entity_weight = pkl.load(f)
    topic_assignments = topic_index_list
    os.chdir('./adj/gossip')
    for item in inputdict.items():
        item = item[1]
        idx = item['idx']
        graph = build_heterogeneous_graph(item, entity_weight, topic_assignments)
        adjlist = extract_adjacency_matrices(graph)
        item['adj_list'] = adjlist
        with open('{}.pkl'.format(idx), 'wb') as f:
            pkl.dump(item, f)