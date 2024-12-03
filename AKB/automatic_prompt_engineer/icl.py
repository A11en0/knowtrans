import pandas as pd
import os
from typing import Literal, Optional
from fuzzywuzzy import fuzz
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split  
import json
import re
import csv
import random
from datetime import datetime
from dateutil import parser
import numpy as np
import copy
from collections import OrderedDict  
from sklearn.metrics import jaccard_score
from scipy.spatial.distance import euclidean
import heapq
from multiprocessing import Pool, Array
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import pickle
from transformers import AutoTokenizer
from functools import partial
import faiss

sbert_model_path = '/share/home/12351018/pre-train/sentence-transformers' #all-MiniLM-L6-v2
sbert_model = None

tokenizer_path = '/share/home/12351018/pre-train/Mistral-7B-OpenOrca'
tokenizer = None

task_type = 'EM'
def set_task_type(tmp):
    global task_type
    task_type = tmp
def read_task_type():
    return task_type

def task_type_mapping():
    task_type_map = {
        'ED': 0,
        'DI': 1,
        'SM': 2,
        'EM': 3,
        "CTA": 4,
        "AVE": 5,
        "DC": 6,
    }
    return task_type_map[read_task_type()]

file_dir = ''
def set_file_dir(tmp):
    global file_dir
    file_dir = tmp
def read_file_dir():
    return file_dir

demo_embeddings = []
demo_pool = []
def set_demo_embeddings(tmp):
    global demo_embeddings
    demo_embeddings = tmp
def read_demo_embeddings():
    global demo_embeddings
    return demo_embeddings
def set_demo_pool(tmp):
    global demo_pool
    demo_pool = tmp
def read_demo_pool():
    global demo_pool
    return demo_pool

query_embeddings = []
def set_query_embeddings(tmp):
    global query_embeddings
    query_embeddings = tmp
def read_query_embeddings():
    global query_embeddings
    return query_embeddings

def read_pickle(file_name):
    file_path = os.path.join(read_file_dir(),file_name+'.pickle')
    if os.path.exists(file_path):
        with open(file_path, 'rb') as file:
            return pickle.load(file)
    else:
        raise ValueError(f"pickle file is not exist: {file_path}")
    
class Faiss:
    def __init__(self):
        self.faiss_index = None

    def faiss_init(self, metric, vectors):
        # 创建索引，这里使用L2距离（欧式距离）
        if metric == 'L2':
            self.faiss_index = faiss.IndexFlatL2(vectors.shape[1])
            self.faiss_index.add(vectors)
        # 创建一个Faissv Flat IP索引（内积 == 余弦相似度 when 向量模长固定时）
        elif metric == 'IP':
            self.faiss_index = faiss.IndexFlatIP(vectors.shape[1])
            self.faiss_index.add(vectors)

    def faiss_query(self, query, k):
        query_vector = np.array(query, dtype=np.float32)
        if len(query_vector.shape) == 1:
            query_vector = query_vector.reshape(1, -1) 
        distances, indices = self.faiss_index.search(query_vector, k)
        return indices[0].tolist()
    
faiss_index = Faiss()


query_patterns = []
def set_query_patterns(tmp):
    global query_patterns
    query_patterns = tmp
def read_query_patterns():
    global query_patterns
    return query_patterns

demo_patterns = []
def set_demo_patterns(tmp):
    global demo_patterns
    demo_patterns = tmp
def read_demo_patterns():
    global demo_patterns
    return demo_patterns


class ClusteringManager:
    def __init__(self):
        self.num_clusters = 0
        self.clustered_sentences = []
        self.cluster_centers = []
        self.cluster_assignment = []
        self.clustered_idx = []
        self.clustered_faiss = []
    
    def set_clustered_sentences(self, clustered_sentences):
        self.clustered_sentences = clustered_sentences
    
    def get_clustered_sentences(self):
        return self.clustered_sentences
    
    def closet_query(self, query, k):
        query = np.array(query, dtype=np.float32)
        # Find k nearest cluster centers to the query vector
        distances_to_centers = np.linalg.norm(self.cluster_centers - query, axis=1)
        closest_k_indices = np.argsort(distances_to_centers)[:k]
        closest_cluster_ids = [self.cluster_assignment[idx] for idx in closest_k_indices]
        return closest_cluster_ids
    
    def clustering(self, num_clusters, corpus, corpus_embeddings):
        self.num_clusters = num_clusters
        # Perform kmean clustering
        model_file = os.path.join(read_file_dir(),f'kmeans_model{num_clusters}.pkl')
        # if os.path.exists(model_file):
        #     print("Loading pre-trained K-means model...")
        #     with open(model_file, 'rb') as f:
        #         clustering_model = pickle.load(f)
        # else:
        clustering_model = KMeans(n_clusters=num_clusters, random_state=42)
        clustering_model.fit(corpus_embeddings)

        self.cluster_assignment = clustering_model.labels_
        self.clustered_sentences = [[] for _ in range(num_clusters)]

        dist = clustering_model.transform(corpus_embeddings)
        clustered_dists = [[] for _ in range(num_clusters)]
        self.clustered_idx = [[] for _ in range(num_clusters)]
        for sentence_id, cluster_id in enumerate(self.cluster_assignment):
            self.clustered_sentences[cluster_id].append(corpus[sentence_id])
            clustered_dists[cluster_id].append(dist[sentence_id][cluster_id])
            self.clustered_idx[cluster_id].append(sentence_id)

        # Get cluster centers
        self.cluster_centers = clustering_model.cluster_centers_

        # 使用 t-SNE 进行降维
        tsne = TSNE(n_components=2, random_state=42)
        embeddings_2d = tsne.fit_transform(corpus_embeddings)

        # 绘制聚类结果的散点图
        plt.figure(figsize=(10, 8))
        for cluster_id in range(num_clusters):
            cluster_points = embeddings_2d[np.array(self.clustered_idx[cluster_id])]
            plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {cluster_id}')
        plt.title('t-SNE Visualization of Clusters')
        plt.xlabel('t-SNE Dimension 1')
        plt.ylabel('t-SNE Dimension 2')
        plt.legend()
        plt.show()
        plt.savefig(os.path.join(read_file_dir(),f"cluster {num_clusters}.png"), dpi=600)

        with open(model_file, 'wb') as f:
            pickle.dump(clustering_model, f)

    def build_faiss_indexes(self, metric, demo_vectors):
        self.clustered_faiss = [Faiss() for _ in range(self.num_clusters)]

        for idx in range(self.num_clusters):
            vectors = [demo_vectors[sentence_id] for sentence_id in self.clustered_idx[idx]]
            vectors = np.array(vectors, dtype=np.float32)
            self.clustered_faiss[idx].faiss_init(metric, vectors)

    def clustered_query(self, idx, query, k):
        return self.clustered_faiss[idx].faiss_query(query,k)
    
clustering_manager = ClusteringManager()

similarity_sentences = []#[[] for _ in range(10000)]
similarity_metric = None
def read_similarity_metric():
    global similarity_metric
    return similarity_metric
def set_similarity_metric(ICL_args):
    if ICL_args is None:
        return
    global similarity_metric
    for ICL_method, ICL_func in ICL_args.items():
        if ICL_method == 'similarity':
            pattern = r'(\w+)-(\d+)'
            match_results = re.findall(pattern, ICL_func)
            metric = match_results[0][0]
            similarity_metric = metric

def read_similarity_sentences(index):
    if read_similarity_metric() is None:
        return 
    
    global similarity_sentences 
    if len(similarity_sentences)==0:
        file_path = os.path.join(read_file_dir(),read_similarity_metric()+'.pickle')
        if os.path.exists(file_path):
            with open(file_path, 'rb') as file:
                similarity_sentences = pickle.load(file)
                print(f"[INFO]Loading pre-saved similarity results {len(similarity_sentences)}...")
        else:
            similarity_sentences = [[] for _ in range(10000)]
    if len(similarity_sentences) <= index:
        raise ValueError(f"[ERROR] {len(similarity_sentences)} is shorter than {index}")
    return similarity_sentences[index]

def set_similarity_sentences(index, content):
    global similarity_metric
    global similarity_sentences
    # print(type(similarity_sentences))
    similarity_sentences[index] = content

def save_similarity_sentences(data):
    if read_similarity_metric() is None:
        return 
    
    file_path = os.path.join(read_file_dir(),similarity_metric+'.pickle')
    if not os.path.exists(file_path) and data is not None:
        print(f"[INFO]Saving {len(data)} similarity results at {file_path}...")
        with open(file_path, 'wb') as file:
            pickle.dump(data, file)


def init_process():
    global file_dir
    global demo_embeddings
    global task_type
    global query_embeddings
    global clustering_manager
    global similarity_metric
    global similarity_sentences

def serialize_(entities):
        table = ""
        for entity in entities:
            row_values = []
            for pair in entity:
                attr_name, attr_value = pair
                if pd.isna(attr_value) or attr_value=="" or attr_value=="?":  
                    row_values.append(f"{attr_name}: \"nan\"")  # 将空值转换为空字符串
                else:
                    row_values.append(f"{attr_name}: \"{attr_value}\"")
            table += "[" + ", ".join(row_values) + "]"
        return table

def serialize_bert(instance):
        table = ""
        pairs_list = []
        for pair in instance['entityA']+instance['entityB'] if read_task_type() in ['EM','SM'] else instance['entity']:
            attr_name, attr_value = pair
            if pd.isna(attr_value) or attr_value=="" or attr_value=="?":  
                pairs_list.append(f"{attr_name}: \"nan\"")  # 将空值转换为空字符串
            else:
                pairs_list.append(f"{attr_name}: \"{attr_value}\"")
        table = "[SEP]".join(pairs_list)
        return table


def levenshtein_distance(s1, s2):
    try:
        if len(s1) < len(s2):
            return levenshtein_distance(s2, s1)
    except:
        print(s1)
        print(type(s1))
        print(np.isnan(s1))
        print(len(s1))
        print(s2)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1 
            deletions = current_row[j] + 1       
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]

def levenshtein_ratio(s1, s2):
    if not isinstance(s1,str) or not isinstance(s2,str):
        # print(f"value is not str: {s1} {s2}")
        return 0
    distance = levenshtein_distance(s1, s2)
    # max_len = max(len(s1), len(s2))
    # return (max_len - distance) / max_len
    sum_len = len(s1)+len(s2)
    if sum_len==0:
        return 0
    return 1 - distance / sum_len

def levenshtein_similarity(list1, list2):
    # 计算Jaccard相似度
    similarities = []
    for item1, item2 in zip(list1, list2):
        # print(item1)
        # print('###')
        # print(item2)
        similarity = levenshtein_ratio(item1[1], item2[1])
        similarities.append(similarity)
    
    # 使用欧氏距离计算距离
    similarities = np.array(similarities)
    distance = euclidean(similarities, np.zeros_like(similarities))
    
    return distance

def jaccard_similarity(list1, list2):
    # 计算Jaccard相似度
    jaccard_similarities = []
    for set1, set2 in zip(list1, list2):
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))

        jaccard_similarity = intersection / union
        # jaccard_similarity = jaccard_score(item1, item2)
        jaccard_similarities.append(jaccard_similarity)
    
    # 使用欧氏距离计算距离
    jaccard_similarities = np.array(jaccard_similarities)
    distance = euclidean(jaccard_similarities, np.zeros_like(jaccard_similarities))
    
    return distance

def jaccard_sim_vector(set_pair):
    sim_v = []
    for set1, set2 in zip(set_pair[0], set_pair[1]):
        intersection = len(set1.intersection(set2))
        # union = len(set1.union(set2))
        union = len(set1)

        jaccard_similarity = intersection / union
        sim_v.append(jaccard_similarity)
    
    return np.array(sim_v)

def levenshtein_vector(list1, list2):
    sim_v = []
    for item1, item2 in zip(list1, list2):
        lr_sim = levenshtein_ratio(item1[1], item2[1])
        sim_v.append(lr_sim)
    
    return np.array(sim_v, dtype=np.float32)

def get_tokens(instances):
    global tokenizer
    token_set = []
    for instance in instances:
        if read_task_type()=='EM' or read_task_type() == 'SM':
            values_listA = [pair[-1] for pair in instance['entityA']]
            values_listB = [pair[-1] for pair in instance['entityB']]
            tokenizered_listA = tokenizer(values_listA)['input_ids']
            tokenizered_listB = tokenizer(values_listB)['input_ids']
            token_set.append(([set(tokens) for tokens in tokenizered_listA], [set(tokens) for tokens in tokenizered_listB]))
        else:
            raise ValueError(f"[DEBUG] 506")
    return token_set

def get_embeddings(sentences):
    # sentences = ["This is an example sentence", "Each sentence is converted"]
    # os.environ['TRANSFORMERS_OFFLINE']='1'
    global sbert_model
    if sbert_model is None:
        sbert_model = SentenceTransformer(sbert_model_path)
    embeddings = sbert_model.encode(sentences)
    return embeddings

def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    return dot_product / (norm_vec1 * norm_vec2)

def embedding_similarity(embedding1, embedding2):
    return cosine_similarity(embedding1, embedding2)

def get_demos(query, ICL_method, self_index=1000000, random_seed=42):    
    # random.seed(random_seed+self_index)
    
    demos = []
    global faiss_index
    demo_pool = read_demo_pool()

    for ICL_method, ICL_func in ICL_method.items():
        if 'sim' in ICL_method:
            pattern = r'(\w+)-(\d+)'
            match_results = re.findall(pattern, ICL_func)
            metric = match_results[0][0]
            k = int(match_results[0][1])
            min_heap = []

            if metric == 'em':
                demos_index = faiss_index.faiss_query(get_embeddings([query]), k)
                demos = [demo_pool[i] for i in demos_index]
                continue
            elif metric == 'lr':
                demos_index = faiss_index.faiss_query(read_query_patterns()[self_index], k)
                demos = [demo_pool[i] for i in demos_index]
                continue

            # saved_demos = read_similarity_sentences(self_index)
            # if len(saved_demos):
            #     demos = saved_demos[:k]
            #     continue

            for i,demo in enumerate(demo_pool):
                if read_task_type()=='EM':
                    flattened_demo = demo['entityA']+demo['entityB']
                    flattened_query = query['entityA']+query['entityB']
                else:
                    flattened_demo = demo['entity']
                    flattened_query = query['entity']

                if metric == 'lr':
                    sim = levenshtein_similarity(flattened_query, flattened_demo)
                elif metric == 'ja':
                    sim = jaccard_similarity(read_query_patterns()[self_index][0], read_demo_patterns()[i][0])
                # elif metric == 'em':
                #     sim = embedding_similarity(get_embeddings(query), read_demo_embeddings()[i])
                else:
                    raise ValueError(f"[ERROR] metric not accepted.(line 585)")

                if len(min_heap) < k: 
                    heapq.heappush(min_heap, (sim, demo))
                    # print(sim, min_heap[0][0])
                elif sim > min_heap[0][0]:
                    try:
                        heapq.heappushpop(min_heap, (sim, demo))
                    except:
                        print(f"[ERROR]heapq push error in line 383: {sim} {demo}")
            
            # try:
                # print(f"demo_sim: {[heapq.heappop(min_heap)[0] for _ in range(len(min_heap))][::-1]}") #
                demos = [heapq.heappop(min_heap)[1] for _ in range(len(min_heap))][::-1] #
                set_similarity_sentences(self_index,demos)
                demos = demos[:k]
            # except:
            #     print(f"[ERROR] 566 {[m for m in min_heap]}")

        elif 'pattern' in ICL_method:
            pattern = r'(\w+)-(\d+)'
            match_results = re.findall(pattern, ICL_func)
            metric = match_results[0][0]
            k = int(match_results[0][1])

            if metric == 'ja':
                pattern_vector = jaccard_sim_vector(read_query_patterns()[self_index])
                demos_index = faiss_index.faiss_query([pattern_vector], k)
                demos = [demo_pool[i] for i in demos_index]
            elif metric == 'lr':
                pattern_vector = read_query_patterns()[self_index]
                demos_index = faiss_index.faiss_query([pattern_vector], k)
                demos = [demo_pool[i] for i in demos_index]

        elif 'div' in ICL_method:
            demos = []
            match_results = re.findall(r'(\w+)-(\d+)', ICL_func)
            metric = match_results[0][0]
            k = int(match_results[0][1])
            global clustering_manager
            cluster_list = clustering_manager.get_clustered_sentences()
            if metric == 'ran':
                # 从cluster_list中随机选择k个聚类的索引
                selected_clusters = random.sample(range(len(cluster_list)), k)

                # 对于每个被选中的聚类，从中随机选择一个值
                for idx in selected_clusters:
                    sampled_value = random.choice(cluster_list[idx])
                    demos.append(sampled_value)
            elif metric == 'em':
                cluster_ids = clustering_manager.closet_query(get_embeddings([query]), k)
                for idx in cluster_ids:
                    sampled_value = random.choice(cluster_list[idx])
                    demos.append(sampled_value)
            elif metric == 'ja':
                cluster_ids = clustering_manager.closet_query(jaccard_sim_vector(read_query_patterns()[self_index]), k)
                for idx in cluster_ids:
                    sampled_value = random.choice(cluster_list[idx])
                    demos.append(sampled_value)
            elif metric == 'emem':
                cluster_ids = clustering_manager.closet_query(get_embeddings(query), k)
                for idx in cluster_ids:
                    demos_index = clustering_manager.clustered_query(idx, get_embeddings(query), 1)
                    demos.append(demo_pool[demos_index[0]])
            elif metric == 'jaem':
                cluster_ids = clustering_manager.closet_query(jaccard_sim_vector(read_query_patterns()[self_index]), k)
                for idx in cluster_ids:
                    demos_index = clustering_manager.clustered_query(idx, get_embeddings(query), 1)
                    demos.append(demo_pool[demos_index[0]])
            elif metric == 'jaja':
                cluster_ids = clustering_manager.closet_query(jaccard_sim_vector(read_query_patterns()[self_index]), k)
                for idx in cluster_ids:
                    demos_index = clustering_manager.clustered_query(idx, jaccard_sim_vector(read_query_patterns()[self_index]), 1)
                    demos.append(demo_pool[demos_index[0]])

        elif 'mix' in ICL_method:
            match_results = re.findall(r'(\w+)-(\d+)', ICL_func)
            metric = match_results[0][0]
            k = int(match_results[0][1])

            cluster_ids = clustering_manager.closet_query(get_embeddings([query]), k)
            for idx in cluster_ids:
                demos_index = clustering_manager.clustered_query(idx, get_embeddings([query]), 1)
                demos.append(demo_pool[demos_index[0]])

        elif 'ran' in ICL_method:
            k = int(ICL_func)
            demos = random.sample(demo_pool,k)

        # elif ICL_method == 'balanced' and read_task_type() == 'EM':
        #     k = int(int(ICL_func) / 2)
        #     positive_demos = [demo for demo in demos if demo[2]==1]
        #     nagetive_demos = [demo for demo in demos if demo[2]==0]
        #     if len(positive_demos) < k :
        #         print(f"[warning] {self_index}'s positive_demos is less than k!")
        #     demos = positive_demos[:min(k, len(positive_demos))] + nagetive_demos[:max(k,2*k-len(positive_demos))]
        else:
            raise ValueError(f"ICL_method is not correct: {ICL_method}")
        
    return demos


def build_faiss(file_path, ICL_method):
    with open(file_path, 'r', encoding='utf-8') as file:
        demo_list = json.load(file)
    embeddings = get_embeddings([demo['input'] for demo in demo_list])
    set_demo_embeddings(embeddings)
    set_demo_pool(demo_list)
    global faiss_index

    method,func  = list(ICL_method.items())[0]
    # match_results = re.findall(r'(\w+)-(\d+)', func)[0]
    # metric = match_results[0]
    # k = int(match_results[1])
    k = 20

    if method == 'sim':
        vectors = np.array(read_demo_embeddings(),dtype=np.float32)
        faiss_index.faiss_init('IP', vectors)
    elif method == 'div' or method == 'mix':
        clustering_manager.clustering(k, demo_list, read_demo_embeddings())
        clustering_manager.build_faiss_indexes('IP', read_demo_embeddings())
    elif method == 'ran':
        pass
    else:
        raise ValueError(f"ICL_method is not correct: {ICL_method}")

