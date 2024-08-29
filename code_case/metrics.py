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

    mean_0_vec = np.mean(class_0_array, axis=1)
    mean_0 = np.mean(mean_0_vec)
    mean_1_vec = np.mean(class_1_array, axis=1)
    mean_1 = np.mean(mean_1_vec)
    stdev_0 = np.std(class_0_array, axis=1)
    stdev_1 = np.std(class_1_array, axis=1)
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

    stdev_0 = np.std(class_0_array, axis=1)
    stdev_1 = np.std(class_1_array, axis=1)
    stdev_mean_0 = np.mean(stdev_0)
    stdev_mean_1 = np.mean(stdev_1)

    avg_stdev = np.mean([stdev_mean_0,stdev_mean_1])

    return avg_stdev

def std_dev(collection):
    # this is class agnostic so just the data is fine
    collection_meta = collection.get(include=["embeddings", "metadatas"])
    embeddings, _labels = vecdb_2_labeled_embeddings(collection_meta)

    print(f"N_chuncked = {len(embeddings)}\n")
    dir = "./dataset/new_test_code"
    files_len = len(os.listdir(dir))
    print(f"N = {len(os.listdir(dir))}\n")

    print(f"avg chunks per file: {len(embeddings)/files_len}\n")

    mean = np.mean(embeddings, axis=1)
    mean_mean = np.mean(mean)
    stdev = np.std(embeddings, axis=1)
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

def construct_diverse_collection(seed_dir, diverse_dir, embedding_function, seed_collection_name, new_collection_name, diversity_amount, chunking=False):
    """
    This function creates a new colllection that is diversified (or appends to an existing one, so be careful about the name)

    Args:
        seed_dir: The directory of the seed data to be diversified
        dirverse_dir: The directory of diverse data to be used to diversify
        embedding_function: The embedding function to use for the creation and querying of the collection
        seed_collection_name: The name of the seed collection, based on the seed data, this will be used to construct a simpler 
        new_collection_name: The name of the collection, will be need to call the collection after it's creation
        diversity_amount: the amount of the diverse data to use, expects full, half, or quarter as inputs
        chunking: If the embedding function requires chunking, such as there is a context window for an LLM embedding function,
                    this defaults to False. 
    Returns: 
        collection: the collection that was created
    """
    # this will overwrite other collections that would have the same name
    persistent_client = chromadb.PersistentClient() # this is using default settings, so imports will also need defaults
    collection = persistent_client.get_or_create_collection(new_collection_name, embedding_function=embedding_function)
    seed_collection = persistent_client.get_or_create_collection(seed_collection_name, embedding_function=embedding_function)

    raw_seed_docs = DirectoryLoader(seed_dir, glob='*.py', loader_cls=TextLoader).load()
    raw_diverse_docs_base = DirectoryLoader(diverse_dir, glob='*.py', loader_cls=TextLoader).load()
    diverse_len = len(raw_diverse_docs_base)

    if diversity_amount == "full":
        raw_diverse_docs = raw_diverse_docs_base

    elif diversity_amount == "half":
        diverse_len = int(len(raw_diverse_docs)/2)
        raw_diverse_docs = raw_diverse_docs_base[:diverse_len]
        
    elif diversity_amount == "quater":
        diverse_len = int(len(raw_diverse_docs)/4)
        raw_diverse_docs = raw_diverse_docs_base[:diverse_len]
    else: 
        print("Error: diversity amount needs to be full, half, or quater")
        return collection

    # now we need to remove a subset of the seed data to be replaced by the diverse data. We will remove the least diverse data
    # will will do this by querying the seed collection by it's own data and grabbing the two most similar entries.
    # these two should be the data point itself, which we ignore, and the closet data point, we then trim reflexive pairs, and 
    # subselect the amount we need given our diversity amount, where we grab those that have the smallest value. 
    results = []
    for i, doc in enumerate(raw_seed_docs):
        result = seed_collection.query(
                query_texts=doc.page_content, # Chroma will embed this for you
                n_results=2 # how many results to return
            )
        results.append((i, result))
    
    results_clone = results

    expanded_results = []
    for entry in results_clone:
        distance = entry[1]['distances'][0][1]
        original_file_idx = entry[0]
        original_source = entry[1]['metadatas'][0][0]
        original_filename = original_source['source']
        nearest_source = entry[1]['metadatas'][0][1]
        nearest_filename = nearest_source['source']
        expanded_results.append((distance, original_file_idx, original_filename, nearest_filename))

    # sort the results by distance, construct list of unique nearest_filenames until reaching the amount needed to be replaced
    expanded_results.sort(key=lambda x: x[0])
    
    del_idxs = []
    for i, ent1 in enumerate(expanded_results):
        for j, ent2 in enumerate(expanded_results):
            if j > i:
                if ent1[2] == ent2[3] and ent1[3] == ent2[2]:
                    del_idxs.append(ent1[2])

        if len(del_idxs) >= diverse_len:
            break

    # now to delete these files from the raw docs we read in
    del_idxs.sort(reverse=True)

    # this might fail depending on the file type of raw_seed_docs
    for idx in del_idxs:
        raw_seed_docs.pop(idx)

    # now back to collection construction
    # chunking cases
    if chunking:
        text_splitter = CharacterTextSplitter(chunk_size=512, chunk_overlap=0)
        seed_docs = text_splitter.split_documents(raw_seed_docs)
        diverse_docs = text_splitter.split_documents(raw_diverse_docs)
    else:
        seed_docs = raw_seed_docs
        diverse_docs = raw_diverse_docs

    # add in the reduced seed data 
    for i, entry in enumerate(seed_docs):
        collection.add(ids=f"{i}", embeddings = embedding_function(entry.page_content), metadatas=entry.metadata, documents=entry.page_content)
    # add in the diverse data
    for i, entry in enumerate(diverse_docs):
        collection.add(ids=f"{i}", embeddings = embedding_function(entry.page_content), metadatas=entry.metadata, documents=entry.page_content)

    return collection

if __name__ == "__main__":
    checkpoint = "Salesforce/codet5p-110m-embedding"
    # need to check that the data is being pulled correctly from the metadata in the vectordb
    # need to check the few shot prompt is staying consistent with the class and overall prompt
    embedding_function_chroma_codet = ChromaCodet5pEmbedding(checkpoint)
    persistent_client_tok = chromadb.PersistentClient() # default settings
    # this gets the collection since it's already present
    collection_seed_tokens = persistent_client_tok.get_collection("seed_code", embedding_function=embedding_function_chroma_codet)
    print("There are", collection_seed_tokens.count(), "in the collection")

    metrics = CollectionStats("seed")
    metrics.get_stats(collection_seed_tokens)

    print(f"metrics.class_mean_vecs: {metrics.class_mean_vecs}\n\nmetrics.class_means: {metrics.class_means}\n\nmetrics.mean_vec: {metrics.mean_vec}\n\nmetrics.mean: {metrics.mean}\n\nmetrics.stdev_vec: {metrics.stdev_vec}\n\nmetrics.stdev: {metrics.stdev}\n\nmetrics.class_stdev_vecs: {metrics.class_stdev_vecs}\n\nmetrics.class_stdevs: {metrics.class_stdevs}\n\nmetrics.agg_stdev: {metrics.agg_stdev}\n\nmetrics.avg_stdev: {metrics.avg_stdev}\n\nmetrics.bimod: {metrics.bimod}\n\nmetrics.skewness_vec: {metrics.skewness_vec}\n\nmetrics.skewness: {metrics.skewness}\n\nmetrics.kurtosis_vec: {metrics.kurtosis_vec}\n\nmetrics.kurtosis: {metrics.kurtosis}")