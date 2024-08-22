# langchain version 0.2, 07/03/2024
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import CharacterTextSplitter
import chromadb
from chromadb import Documents, EmbeddingFunction

from langchain_openai import ChatOpenAI
from langchain.pydantic_v1 import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate

import os
from transformers import AutoModel, AutoTokenizer
from typing import List
import random as rd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from scipy.stats import skew, kurtosis


from tree_encoder_model import GNNModel
from code_preprocessing import preprocess_tree_query
import torch

from vector_db import ChromaCodet5pEmbedding, ChromaTreeEmbedding

from tree_model_trainer import EMBED_DIM, IN_CHANNELS, HIDDEN_CHANNELS, OUT_CHANNELS, NODE_CLASSES, COMPRESSED_CLASSES, COMPRESSED_GRAPH_FEATURE, GRAPH_FEATURE

class CollectionStats:
    def __init__(self, collection_name):
        self.collection_name = collection_name
        self.class_mean_vecs = []
        self.class_means = []
        self.mean_vec = []
        self.mean = 0
        self.class_stdev_vecs = []
        self.class_stdevs = []
        self.stdev_vec = []
        self.stdev = 0
        self.agg_stdev = 0
        self.avg_stdev = 0
        self.bimod = 0
        self.skewness_vec = []
        self.skewness = 0
        self.kurtosis_vec = []
        self.kurtosis = 0

    def get_stats(self, collection):
        means, mean_vecs, stdevs, stdev_means, agg_stdev = agg_std_dev(collection)
        avg_stdev = avg_std_dev(collection)
        mean, mean_mean, stdev, stdev_mean = std_dev(collection)
        g, g_mean, k, k_mean, bimod = bimodality(collection)

        self.class_mean_vecs = mean_vecs
        self.class_means = means
        self.mean_vec = mean
        self.mean = mean_mean
        self.stdev_vec = stdev
        self.stdev = stdev_mean
        self.class_stdev_vecs = stdevs
        self.class_stdevs = stdev_means
        self.agg_stdev = agg_stdev
        self.avg_stdev = avg_stdev
        self.bimod = bimod
        self.skewness_vec = g
        self.skewness = g_mean
        self.kurtosis_vec = k
        self.kurtosis = k_mean        


def vecdb_2_labeled_embeddings(dict):
    embeddings = []
    labels = []
    for i, metadata in enumerate(dict['metadatas']):
        embedding = dict['embeddings'][i]
        file = metadata['source']
        label = 0
        if file.split("-")[-2] == "abm":
            label = 1
        
        embeddings.append(embedding)
        labels.append(label)
    
    np_data = np.array(embeddings)
    np_labels = np.array(labels)

    return np_data, np_labels


def agg_std_dev(collection):
    # this is class-specific so we need to read in the class too
    collection_meta = collection.get(include=["embeddings", "metadatas"])
    embeddings, labels = vecdb_2_labeled_embeddings(collection_meta)
    
    class_0 = []
    class_1 = []
    for i, label in enumerate(labels):
        if label == 0:
            class_0.append(embeddings[i])
        else:
            class_1.append(embeddings[i])
    
    class_0_array = np.array(class_0)
    class_1_array = np.array(class_1)

    means = []
    mean_vecs = []
    stdevs = []
    stdev_means = []

    mean_0_vec = np.mean(class_0_array)
    mean_0 = np.mean(mean_0_vec)
    mean_1_vec = np.mean(class_1_array)
    mean_1 = np.mean(mean_1_vec)
    stdev_0 = np.std(class_0_array, axis=0)
    stdev_1 = np.std(class_1_array, axis=0)
    stdev_mean_0 = np.mean(stdev_0)
    stdev_mean_1 = np.mean(stdev_1)

    means.append(mean_0)
    means.append(mean_1)
    mean_vecs.append(mean_0_vec)
    mean_vecs.append(mean_1_vec)
    stdevs.append(stdev_0)
    stdevs.append(stdev_1)
    stdev_means.append(stdev_mean_0)
    stdev_means.append(stdev_mean_1)

    agg_stdev = stdev_mean_0 + stdev_mean_1

    return (means, mean_vecs, stdevs, stdev_means, agg_stdev)

def avg_std_dev(collection):
    # this is class-specific so we need to read in the class too
    collection_meta = collection.get(include=["embeddings", "metadatas"])
    embeddings, labels = vecdb_2_labeled_embeddings(collection_meta)

    class_0 = []
    class_1 = []
    for i, label in enumerate(labels):
        if label == 0:
            class_0.append(embeddings[i])
        else:
            class_1.append(embeddings[i])
    
    class_0_array = np.array(class_0)
    class_1_array = np.array(class_1)

    stdev_0 = np.std(class_0_array, axis=0)
    stdev_1 = np.std(class_1_array, axis=0)
    stdev_mean_0 = np.mean(stdev_0)
    stdev_mean_1 = np.mean(stdev_1)

    avg_stdev = np.mean([stdev_mean_0,stdev_mean_1])

    return avg_stdev

def std_dev(collection):
    # this is class agnostic so just the data is fine
    collection_meta = collection.get(include=["embeddings", "metadatas"])
    embeddings, _labels = vecdb_2_labeled_embeddings(collection_meta)

    mean = np.mean(embeddings)
    mean_mean = np.mean(mean)
    stdev = np.std(embeddings, axis=0)
    stdev_mean = np.mean(stdev)

    return (mean, mean_mean, stdev, stdev_mean)


def bimodality(collection):
    # this is class agnostic so just the data is fine
    collection_meta = collection.get(include=["embeddings", "metadatas"])
    embeddings, _labels = vecdb_2_labeled_embeddings(collection_meta)

    n = len(embeddings)
    g = skew(embeddings)
    g_mean = np.mean(g)
    k = kurtosis(embeddings)
    k_mean = np.mean(k)

    bimod = (g_mean**2 + 1)/(k_mean + (3*(n-1)**2)/((n-2)*(n-3)))

    return (g, g_mean, k, k_mean, bimod)

if __name__ == "__main__":
    # need to check that the data is being pulled correctly from the metadata in the vectordb
    # need to check the few shot prompt is staying consistent with the class and overall prompt
    print("temp")