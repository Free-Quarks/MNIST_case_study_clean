# langchain version 0.2, 07/03/2024
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import CharacterTextSplitter
import chromadb
from chromadb import Documents, EmbeddingFunction
from torch_geometric.loader import DataLoader as GeometricLoader

from langchain_openai import ChatOpenAI
from langchain.pydantic_v1 import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate

from classification_models import TokenClassificationModel, TreeClassificationModel, TokenClassificationModel2
from tree_model_trainer import EMBED_DIM, IN_CHANNELS, HIDDEN_CHANNELS, OUT_CHANNELS, NODE_CLASSES, COMPRESSED_CLASSES, COMPRESSED_GRAPH_FEATURE, GRAPH_FEATURE
from token_classification_trainer import preprocess_tokenized_dataset, TokenDatasetWrapper
from code_preprocessing import preprocess_tree_query

from torch import nn, optim
from torch.utils.data import DataLoader
import torch.nn.functional as F

from sentence_transformers import SentenceTransformer

import os
from transformers import AutoModel, AutoTokenizer
from typing import List
import random as rd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from scipy.stats import skew, kurtosis


from tree_encoder_model import GNNModel
import torch
import pickle

from vector_db import ChromaCodet5pEmbedding, ChromaTreeEmbedding, ChromaNomicEmbedding

from tree_model_trainer import EMBED_DIM, IN_CHANNELS, HIDDEN_CHANNELS, OUT_CHANNELS, NODE_CLASSES, COMPRESSED_CLASSES, COMPRESSED_GRAPH_FEATURE, GRAPH_FEATURE

TEST_CODE = False
GENERATE_TOKEN_DBS = False
GENERATE_TREE_DBS = False
GENERATE_TOKEN_NOMIC_DBS = False
TOKEN_NOMIC_MODELS = False
TOKEN_METRICS = True
TOKEN_MODELS = True
TREE_METRICS = False
TREE_MODELS = False
CONSTRUCT_NOMIC = False

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


def clean_tree_dataset(dataset):
    # need to test and fix the tree data some before putting in loader
    counter = len(dataset) - 1
    for (entry, _label) in reversed(dataset):
        if entry.pe.shape[1] != 12:
            dataset.delete_entry(counter)
        counter += -1
    return dataset

def preprocess_tree_collection_dataset(collection, url, model_checkpoint):
    """
    This converts a vectordb collection into a dataset of encoded tree graphs
    """
    collection_dict = collection.get(include=["embeddings", "metadatas"])
    raw_data = []
    for metadata in collection_dict['metadatas']:
        file_name = metadata['source']
        label = 0
        if file_name.split("-")[-2] == "abm":
            label = 1
        raw_data.append((file_name, label))

    tree_data = []
    counter = 1
    for file_name, label in raw_data:
        with open(file_name, 'r', encoding='utf-8') as file:
            content = file.read()
            file.close()
        encoded_graph = preprocess_tree_query(content, url, model_checkpoint)
        if encoded_graph is not None:
            if len(encoded_graph) == 1:
                tree_data.append((encoded_graph[0], label))
        print(f"file {counter} of {len(raw_data)} done")
        counter += 1
    
    dataset = TokenDatasetWrapper(tree_data)
    return dataset

def preprocess_tokenized_dataset_collection(collection, checkpoint):
    """
    This function takes in a chroma collection of the code that will be processed into a dataset of tokenized sequence inputs.
    It then gets the embedding vector for the code and creates a labeled dataset by processing the label in the filename.

    Args:
        collection (collection): collction of code to be processed
        checkpoint (string): checkpoint for the model we call that does the tokenization
    Returns:
        dataset (TokenDatasetWrapper): a dataset of processed code
    """
    # initialize encoder for tokenization
    device = "cuda:1"  # for GPU usage or "cpu" for CPU usage
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
    model = AutoModel.from_pretrained(checkpoint, trust_remote_code=True).to(device)

    # first we read in and encode all the files in the collection, we need 
    collection_dict = collection.get(include=["embeddings", "metadatas"])
    raw_data = []
    for metadata in collection_dict['metadatas']:
        file_name = metadata['source']
        label = 0
        if file_name.split("-")[-2] == "abm":
            label = 1
        raw_data.append((file_name, label))

    token_data = []
    counter = 1
    for file_name, label in raw_data:
        with open(file_name, 'r') as file:
            content = file.read()
            file.close()

        file_encoding = tokenizer.encode(content, truncation=True, return_tensors="pt").to(device)
        embedding = model(file_encoding).to("cpu").detach()
        token_data.append((embedding, label))
        
        print(f"file {counter} of {len(raw_data)} done")
        counter += 1
            
    # wrap our list into a dataset
    dataset = TokenDatasetWrapper(token_data)
    return dataset

def preprocess_tokenized_dataset_collection_nomic(collection, checkpoint, matryoshka_dim):
    """
    This function takes in a chroma collection of the code that will be processed into a dataset of tokenized sequence inputs.
    It then gets the embedding vector for the code and creates a labeled dataset by processing the label in the filename.

    Args:
        collection (collection): collction of code to be processed
        checkpoint (string): checkpoint for the model we call that does the tokenization
        matryoshka_dim (int): The dimension for the matryoshka embedding vector
    Returns:
        dataset (TokenDatasetWrapper): a dataset of processed code
    """
    # initialize encoder for tokenization
    device = "cuda:1"  # for GPU usage or "cpu" for CPU usage
    model = SentenceTransformer(checkpoint, trust_remote_code=True)

    # first we read in and encode all the files in the collection, we need 
    collection_dict = collection.get(include=["embeddings", "metadatas"])
    raw_data = []
    for metadata in collection_dict['metadatas']:
        file_name = metadata['source']
        label = 0
        if file_name.split("-")[-2] == "abm":
            label = 1
        raw_data.append((file_name, label))

    token_data = []
    counter = 1
    for file_name, label in raw_data:
        with open(file_name, 'r') as file:
            content = file.read()
            file.close()

        embedding = model.encode(content, convert_to_tensor=True)
        #print(embeddings.shape)
        #embeddings = F.layer_norm(embeddings, normalized_shape=(embeddings.shape[0],))
        embedding = embedding[:matryoshka_dim]
        embedding = F.normalize(embedding, p=2, dim=0).to("cpu").detach()
        #embedding = model(file_encoding).to("cpu").detach()
        token_data.append((embedding, label))
        
        print(f"file {counter} of {len(raw_data)} done")
        counter += 1
            
    # wrap our list into a dataset
    dataset = TokenDatasetWrapper(token_data)
    return dataset

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
        diverse_len = int(len(raw_diverse_docs_base)/2)
        raw_diverse_docs = raw_diverse_docs_base[:diverse_len]
        
    elif diversity_amount == "quarter":
        diverse_len = int(len(raw_diverse_docs_base)/4)
        raw_diverse_docs = raw_diverse_docs_base[:diverse_len]
    else: 
        print("Error: diversity amount needs to be full, half, or quarter")
        return collection

    # now we need to remove a subset of the seed data to be replaced by the diverse data. We will remove the least diverse data
    # will will do this by querying the seed collection by it's own data and grabbing the two most similar entries.
    # these two should be the data point itself, which we ignore, and the closet data point, we then trim reflexive pairs, and 
    # subselect the amount we need given our diversity amount, where we grab those that have the smallest value. 
    print("collecting results...")
    if os.path.exists(seed_dir+'/results_'+seed_collection_name+'.pkl'):
        with open(seed_dir+'/results_'+seed_collection_name+'.pkl', 'rb') as file:
            expanded_results = pickle.load(file)
    else:
        result_counter = 0
        results = []
        for i, doc in enumerate(raw_seed_docs):
            result = seed_collection.query(
                    query_texts=doc.page_content, # Chroma will embed this for you
                    n_results=2 # how many results to return
                )
            results.append((i, result))
            result_counter += 1
            print(f"{result_counter}/{len(raw_seed_docs)} results done")
        
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

        # now to save this result so we don't have to compute it each time
        with open(seed_dir+'/results_'+seed_collection_name+'.pkl', 'wb') as file:
            pickle.dump(expanded_results, file)

    # sort the results by distance, construct list of unique nearest_filenames until reaching the amount needed to be replaced
    expanded_results.sort(key=lambda x: x[0])
    print("determining indexes to delete...")
    del_idxs = []
    for i, ent1 in enumerate(expanded_results):
        for j, ent2 in enumerate(expanded_results):
            if j > i:
                if ent1[2] == ent2[3] and ent1[3] == ent2[2]:
                    del_idxs.append(ent1[1])

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
    print("adding trimmed seed docs...")
    i_count = 0
    for i, entry in enumerate(seed_docs):
        collection.add(ids=f"{i}", embeddings = embedding_function(entry.page_content), metadatas=entry.metadata, documents=entry.page_content)
        i_count += 1
    # add in the diverse data
    print("adding diverse docs...")
    for i, entry in enumerate(diverse_docs):
        collection.add(ids=f"{i+i_count+1}", embeddings = embedding_function(entry.page_content), metadatas=entry.metadata, documents=entry.page_content)

    return collection

if __name__ == "__main__":
    if TEST_CODE:
        checkpoint = "Salesforce/codet5p-110m-embedding"
        # need to check that the data is being pulled correctly from the metadata in the vectordb
        # need to check the few shot prompt is staying consistent with the class and overall prompt
        embedding_function_chroma_codet = ChromaCodet5pEmbedding(checkpoint)
        persistent_client_tok = chromadb.PersistentClient() # default settings
        # this gets the collection since it's already present
        collection_seed_tokens = persistent_client_tok.get_collection("seed_code", embedding_function=embedding_function_chroma_codet)

        tree_checkpoint = "./models/tree_ae/tree_ae.pth"
        url = "http://localhost:8000/code2fn/fn-given-filepaths"
        embedding_function_chroma_tree = ChromaTreeEmbedding(checkpoint, tree_checkpoint, url)
        persistent_client_tree = chromadb.PersistentClient() # default settings
        # this gets the collection since it's already present
        collection_seed_tree = persistent_client_tree.get_collection("seed_code_tree", embedding_function=embedding_function_chroma_tree)

        metrics = CollectionStats("seed")
        metrics.get_stats(collection_seed_tokens)

        temp1 = f"Token_seed:\n\nmetrics.stdev: {metrics.stdev:.4f}\n\nmetrics.agg_stdev: {metrics.agg_stdev:.4f}\n\nmetrics.avg_stdev: {metrics.avg_stdev:.4f}\n\nmetrics.bimod: {metrics.bimod:.4f}\n\nmetrics.skewness: {metrics.skewness:.4f}\n\nmetrics.kurtosis: {metrics.kurtosis:.4f}"

        metrics.get_stats(collection_seed_tree)

        temp2 = f"Tree_seed:\n\nmetrics.stdev: {metrics.stdev:.4f}\n\nmetrics.agg_stdev: {metrics.agg_stdev:.4f}\n\nmetrics.avg_stdev: {metrics.avg_stdev:.4f}\n\nmetrics.bimod: {metrics.bimod:.4f}\n\nmetrics.skewness: {metrics.skewness:.4f}\n\nmetrics.kurtosis: {metrics.kurtosis:.4f}"

        print("------------------------------")
        print(temp1)
        print("------------------------------")
        print(temp2)
        print("------------------------------")

        # now to test the creation of a new vectordb
        #collection2 = construct_diverse_collection("./dataset/new_test_code", "./dataset/test_code", embedding_function_chroma_codet, "seed_code", "test_collection2", "half", chunking=True)

        #test_metrics = CollectionStats("test2")
        #test_metrics.get_stats(collection2)

        #print(f"metrics.class_mean_vecs: {test_metrics.class_mean_vecs}\n\nmetrics.class_means: {test_metrics.class_means}\n\nmetrics.mean_vec: {test_metrics.mean_vec}\n\nmetrics.mean: {test_metrics.mean}\n\nmetrics.stdev_vec: {test_metrics.stdev_vec}\n\nmetrics.stdev: {test_metrics.stdev}\n\nmetrics.class_stdev_vecs: {test_metrics.class_stdev_vecs}\n\nmetrics.class_stdevs: {test_metrics.class_stdevs}\n\nmetrics.agg_stdev: {test_metrics.agg_stdev}\n\nmetrics.avg_stdev: {test_metrics.avg_stdev}\n\nmetrics.bimod: {test_metrics.bimod}\n\nmetrics.skewness_vec: {test_metrics.skewness_vec}\n\nmetrics.skewness: {test_metrics.skewness}\n\nmetrics.kurtosis_vec: {test_metrics.kurtosis_vec}\n\nmetrics.kurtosis: {test_metrics.kurtosis}")
    
    if GENERATE_TOKEN_DBS:
        checkpoint = "Salesforce/codet5p-110m-embedding"
        embedding_function = ChromaCodet5pEmbedding(checkpoint)

        seed_dir = "./dataset/new_test_code"
        seed_collection_name = "seed_code"

        token_class_few_dir = "./dataset/new_generator/token_class_few_full"
        token_classless_few_dir = "./dataset/new_generator/token_classless_few_full"
        token_class_zero_dir = "./dataset/new_generator/token_class_zero_full"

        token_class_few_full_name = "token_class_few_full"
        token_class_few_half_name = "token_class_few_half"
        token_class_few_quarter_name = "token_class_few_quarter"
        token_class_zero_name = "token_class_zero"
        token_classless_few_name = "token_classless_few"

        token_class_few_full_collection = construct_diverse_collection(seed_dir, token_class_few_dir, embedding_function, seed_collection_name, token_class_few_full_name, "full", chunking=True)

        token_class_few_half_collection = construct_diverse_collection(seed_dir, token_class_few_dir, embedding_function, seed_collection_name, token_class_few_half_name, "half", chunking=True)

        token_class_few_quarter_collection = construct_diverse_collection(seed_dir, token_class_few_dir, embedding_function, seed_collection_name, token_class_few_quarter_name, "quarter", chunking=True)

        token_class_zero_collection = construct_diverse_collection(seed_dir, token_class_zero_dir, embedding_function, seed_collection_name, token_class_zero_name, "full", chunking=True)

        token_classless_few_collection = construct_diverse_collection(seed_dir, token_classless_few_dir, embedding_function, seed_collection_name, token_classless_few_name, "full", chunking=True)
    
    if GENERATE_TREE_DBS:
        checkpoint = "Salesforce/codet5p-110m-embedding"
        tree_checkpoint = "./models/tree_ae/tree_ae.pth"
        url = "http://localhost:8000/code2fn/fn-given-filepaths"
        embedding_function = ChromaTreeEmbedding(checkpoint, tree_checkpoint, url)

        seed_dir = "./dataset/new_test_code"
        seed_collection_name = "seed_code_tree"

        tree_class_few_dir = "./dataset/new_generator/tree_class_few"
        tree_classless_few_dir = "./dataset/new_generator/tree_classless_few"
        tree_class_zero_dir = "./dataset/new_generator/tree_class_zero"

        tree_class_few_full_name = "tree_class_few_full3"
        tree_class_few_half_name = "tree_class_few_half3"
        tree_class_few_quarter_name = "tree_class_few_quarter3"
        tree_class_zero_name = "tree_class_zero3"
        tree_classless_few_name = "tree_classless_few3"

        tree_class_few_full_collection = construct_diverse_collection(seed_dir, tree_class_few_dir, embedding_function, seed_collection_name, tree_class_few_full_name, "full", chunking=False)

        tree_class_few_half_collection = construct_diverse_collection(seed_dir, tree_class_few_dir, embedding_function, seed_collection_name, tree_class_few_half_name, "half", chunking=False)

        tree_class_few_quarter_collection = construct_diverse_collection(seed_dir, tree_class_few_dir, embedding_function, seed_collection_name, tree_class_few_quarter_name, "quarter", chunking=False)

        tree_class_zero_collection = construct_diverse_collection(seed_dir, tree_class_zero_dir, embedding_function, seed_collection_name, tree_class_zero_name, "full", chunking=False)

        tree_classless_few_collection = construct_diverse_collection(seed_dir, tree_classless_few_dir, embedding_function, seed_collection_name, tree_classless_few_name, "full", chunking=False)

    if GENERATE_TOKEN_NOMIC_DBS:
        checkpoint = "nomic-ai/nomic-embed-text-v1.5"

        matryoshka_dim = 128
        embedding_function = ChromaNomicEmbedding(checkpoint, matryoshka_dim)

        seed_dir = "./dataset/new_test_code"
        seed_collection_name = "seed_code_nomic"

        token_class_few_dir = "./dataset/new_generator/token_nomic_class_few"
        token_classless_few_dir = "./dataset/new_generator/token_nomic_classless_few"
        token_class_zero_dir = "./dataset/new_generator/token_nomic_class_zero"

        token_class_few_full_name = "token_nomic_class_few_full2"
        token_class_few_half_name = "token_nomic_class_few_half2"
        token_class_few_quarter_name = "token_nomic_class_few_quarter2"
        token_class_zero_name = "token_nomic_class_zero2"
        token_classless_few_name = "token_nomic_classless_few2"

        if CONSTRUCT_NOMIC:
            token_class_few_full_collection = construct_diverse_collection(seed_dir, token_class_few_dir, embedding_function, seed_collection_name, token_class_few_full_name, "full", chunking=False)

            token_class_few_half_collection = construct_diverse_collection(seed_dir, token_class_few_dir, embedding_function, seed_collection_name, token_class_few_half_name, "half", chunking=False)

            token_class_few_quarter_collection = construct_diverse_collection(seed_dir, token_class_few_dir, embedding_function, seed_collection_name, token_class_few_quarter_name, "quarter", chunking=False)

            token_class_zero_collection = construct_diverse_collection(seed_dir, token_class_zero_dir, embedding_function, seed_collection_name, token_class_zero_name, "full", chunking=False)

            token_classless_few_collection = construct_diverse_collection(seed_dir, token_classless_few_dir, embedding_function, seed_collection_name, token_classless_few_name, "full", chunking=False)

        else:
            persistent_client_few_full = chromadb.PersistentClient()
            token_class_few_full_collection = persistent_client_few_full.get_collection(token_class_few_full_name, embedding_function=embedding_function)

            persistent_client_few_half = chromadb.PersistentClient()
            token_class_few_half_collection = persistent_client_few_half.get_collection(token_class_few_half_name, embedding_function=embedding_function)

            persistent_client_few_quarter = chromadb.PersistentClient()
            token_class_few_quarter_collection = persistent_client_few_half.get_collection(token_class_few_quarter_name, embedding_function=embedding_function)

            persistent_client_zero = chromadb.PersistentClient()
            token_class_zero_collection = persistent_client_zero.get_collection(token_class_zero_name, embedding_function=embedding_function)

            persistent_client_classless = chromadb.PersistentClient()
            token_classless_few_collection = persistent_client_classless.get_collection(token_classless_few_name, embedding_function=embedding_function)

        persistent_client_seed = chromadb.PersistentClient()
        seed_nomic_collection = persistent_client_seed.get_collection(seed_collection_name, embedding_function=embedding_function)

        test_metrics = CollectionStats("test_2")
        test_metrics.get_stats(token_class_few_full_collection)
        temp1 = f"Token_few_full:\n\nmetrics.stdev: {test_metrics.stdev:.4f}\n\nmetrics.agg_stdev: {test_metrics.agg_stdev:.4f}\n\nmetrics.avg_stdev: {test_metrics.avg_stdev:.4f}\n\nmetrics.bimod: {test_metrics.bimod:.4f}\n\nmetrics.skewness: {test_metrics.skewness:.4f}\n\nmetrics.kurtosis: {test_metrics.kurtosis:.4f}"
        test_metrics.get_stats(token_class_few_half_collection)
        temp2 = f"Token_few_half:\n\nmetrics.stdev: {test_metrics.stdev:.4f}\n\nmetrics.agg_stdev: {test_metrics.agg_stdev:.4f}\n\nmetrics.avg_stdev: {test_metrics.avg_stdev:.4f}\n\nmetrics.bimod: {test_metrics.bimod:.4f}\n\nmetrics.skewness: {test_metrics.skewness:.4f}\n\nmetrics.kurtosis: {test_metrics.kurtosis:.4f}"
        test_metrics.get_stats(token_class_few_quarter_collection)
        temp3 = f"Token_few_quarter:\n\nmetrics.stdev: {test_metrics.stdev:.4f}\n\nmetrics.agg_stdev: {test_metrics.agg_stdev:.4f}\n\nmetrics.avg_stdev: {test_metrics.avg_stdev:.4f}\n\nmetrics.bimod: {test_metrics.bimod:.4f}\n\nmetrics.skewness: {test_metrics.skewness:.4f}\n\nmetrics.kurtosis: {test_metrics.kurtosis:.4f}"
        test_metrics.get_stats(token_class_zero_collection)
        temp4 =  f"Token_zero:\n\nmetrics.stdev: {test_metrics.stdev:.4f}\n\nmetrics.agg_stdev: {test_metrics.agg_stdev:.4f}\n\nmetrics.avg_stdev: {test_metrics.avg_stdev:.4f}\n\nmetrics.bimod: {test_metrics.bimod:.4f}\n\nmetrics.skewness: {test_metrics.skewness:.4f}\n\nmetrics.kurtosis: {test_metrics.kurtosis:.4f}"
        test_metrics.get_stats(token_classless_few_collection)
        temp5 =  f"Token_few_classless:\n\nmetrics.stdev: {test_metrics.stdev:.4f}\n\nmetrics.agg_stdev: {test_metrics.agg_stdev:.4f}\n\nmetrics.avg_stdev: {test_metrics.avg_stdev:.4f}\n\nmetrics.bimod: {test_metrics.bimod:.4f}\n\nmetrics.skewness: {test_metrics.skewness:.4f}\n\nmetrics.kurtosis: {test_metrics.kurtosis:.4f}"

        print("------------------------------")
        print(temp1)
        print("------------------------------")
        print(temp2)
        print("------------------------------")
        print(temp3)
        print("------------------------------")
        print(temp4)
        print("------------------------------")
        print(temp5)
        print("------------------------------")

        if TOKEN_NOMIC_MODELS:
            checkpoint = "nomic-ai/nomic-embed-text-v1.5"
            matryoshka_dim = 128
            # encoded dataset locations
            seed_nomic = "./dataset/encoded_tokens_nomic/seed_tokens_nomic.pth"
            class_few_full_path = "./dataset/encoded_tokens_nomic/class_few_full.pth"
            class_few_half_path = "./dataset/encoded_tokens_nomic/class_few_half.pth"
            class_few_quarter_path = "./dataset/encoded_tokens_nomic/class_few_quarter.pth"
            class_zero_path = "./dataset/encoded_tokens_nomic/class_zero.pth"
            classless_few_path = "./dataset/encoded_tokens_nomic/classless_few.pth"

            if os.path.exists(seed_nomic):
                seed_nomic_dataset = torch.load(seed_nomic)
            else:
                seed_nomic_dataset = preprocess_tokenized_dataset_collection_nomic(seed_nomic_collection, checkpoint, matryoshka_dim)
                torch.save(seed_nomic_dataset, seed_nomic)

            if os.path.exists(class_few_full_path):
                token_class_few_full_dataset = torch.load(class_few_full_path)
            else:
                token_class_few_full_dataset = preprocess_tokenized_dataset_collection_nomic(token_class_few_full_collection, checkpoint, matryoshka_dim)
                torch.save(token_class_few_full_dataset, class_few_full_path)
            if os.path.exists(class_few_half_path):
                token_class_few_half_dataset = torch.load(class_few_half_path)
            else:
                token_class_few_half_dataset = preprocess_tokenized_dataset_collection_nomic(token_class_few_half_collection, checkpoint, matryoshka_dim)
                torch.save(token_class_few_half_dataset, class_few_half_path)
            if os.path.exists(class_few_quarter_path):
                token_class_few_quarter_dataset = torch.load(class_few_quarter_path)
            else:
                token_class_few_quarter_dataset = preprocess_tokenized_dataset_collection_nomic(token_class_few_quarter_collection, checkpoint, matryoshka_dim)
                torch.save(token_class_few_quarter_dataset, class_few_quarter_path)
            if os.path.exists(class_zero_path):
                token_class_zero_dataset = torch.load(class_zero_path)
            else:
                token_class_zero_dataset = preprocess_tokenized_dataset_collection_nomic(token_class_zero_collection, checkpoint, matryoshka_dim)
                torch.save(token_class_zero_dataset, class_zero_path)
            if os.path.exists(classless_few_path):
                token_classless_few_dataset = torch.load(classless_few_path)
            else:
                token_classless_few_dataset = preprocess_tokenized_dataset_collection_nomic(token_classless_few_collection, checkpoint, matryoshka_dim)
                torch.save(token_classless_few_dataset, classless_few_path)

            #load_seed_dataset_tree = torch.load('./dataset/encoded_tokens/seed_tokens.pth')

            # configuration values here:
            LR_RATE = 3e-4
            BATCH_SIZE = 32
            MAX_EPOCHS = 2
            DEVICE = "cuda:1"
            NUM_RUNS = 3

            # model values here
            TOK_EMBED_DIM = 128
            TOK_IN_CHANNELS = 72
            TOK_HIDDEN_LAY1 = 36
            TOK_HIDDEN_LAY2 = 18
            GRAPH_CLASS = 1

            # now to get all the datasets into datalaoders
            loader_seed_dataset_tree = DataLoader(dataset=seed_nomic_dataset, batch_size=BATCH_SIZE, shuffle=True)
            loader_tree_class_few_full_dataset = DataLoader(dataset=token_class_few_full_dataset, batch_size=BATCH_SIZE, shuffle=True)
            loader_tree_class_few_half_dataset = DataLoader(dataset=token_class_few_half_dataset, batch_size=BATCH_SIZE, shuffle=True)
            loader_tree_class_few_quarter_dataset = DataLoader(dataset=token_class_few_quarter_dataset, batch_size=BATCH_SIZE, shuffle=True)
            loader_tree_class_zero_dataset = DataLoader(dataset=token_class_zero_dataset, batch_size=BATCH_SIZE, shuffle=True)
            loader_tree_classless_few_dataset = DataLoader(dataset=token_classless_few_dataset, batch_size=BATCH_SIZE, shuffle=True)

            # to make it easier to iterate over all the dataset, we will add them into a list now, with a name for the second entry in the tuple
            loader_list = []
            loader_list.append((loader_seed_dataset_tree, "seed_token"))
            loader_list.append((loader_tree_class_few_full_dataset, "class_few_full_token"))
            loader_list.append((loader_tree_class_few_half_dataset, "class_few_half_token"))
            loader_list.append((loader_tree_class_few_quarter_dataset, "class_few_quarter_token"))
            loader_list.append((loader_tree_class_zero_dataset, "class_zero_token"))
            loader_list.append((loader_tree_classless_few_dataset, "classless_few_token"))

            # now to setup a model, train it, get the metrics, save the model, and then reset everything for the next dataset
            for run in range(NUM_RUNS):
                for (loader, name) in loader_list:
                    model = TokenClassificationModel(TOK_EMBED_DIM, TOK_IN_CHANNELS, TOK_HIDDEN_LAY1, TOK_HIDDEN_LAY2, GRAPH_CLASS).to(DEVICE)
                    optimizer = optim.Adam(model.parameters(), lr=LR_RATE) # adam optimizer
                    loss_function = nn.BCELoss()
                    model.train()

                    for epoch in range(1, MAX_EPOCHS):
                        overall_loss = 0
                        for batch_idx, (data, label) in enumerate(loader):
                            data = data.to(DEVICE)
                            label = label.to(DEVICE)
                            optimizer.zero_grad()
                            output = model(data)
                            loss = loss_function(output.view(label.shape[0]), label.float())
                            overall_loss += loss.item()
                            loss.backward()
                            optimizer.step()
                        print("\tSeed Epoch", epoch, "\tSeed Average Loss: ", overall_loss/((batch_idx+1)*BATCH_SIZE))

                    # now to save the model 
                    torch.save(model.state_dict(), f"./models/nomic_token_models/{name}/model_{run}.pth")
                    del model


    if TOKEN_METRICS:
        token_class_few_full_name = "token_class_few_full"
        token_class_few_half_name = "token_class_few_half"
        token_class_few_quarter_name = "token_class_few_quarter"
        token_class_zero_name = "token_class_zero"
        token_classless_few_name = "token_classless_few"

        checkpoint = "Salesforce/codet5p-110m-embedding"
        embedding_function = ChromaCodet5pEmbedding(checkpoint)

        # spool up 5 clients
        persistent_client_class_few_full = chromadb.PersistentClient()
        persistent_client_class_few_half = chromadb.PersistentClient()
        persistent_client_class_few_quarter = chromadb.PersistentClient()
        persistent_client_class_zero = chromadb.PersistentClient()
        persistent_client_classless_few = chromadb.PersistentClient()


        # grab the collections
        token_class_few_full = persistent_client_class_few_full.get_collection(token_class_few_full_name, embedding_function=embedding_function)

        token_class_few_half = persistent_client_class_few_half.get_collection(token_class_few_half_name, embedding_function=embedding_function)

        token_class_few_quarter = persistent_client_class_few_quarter.get_collection(token_class_few_quarter_name, embedding_function=embedding_function)

        token_class_zero = persistent_client_class_zero.get_collection(token_class_zero_name, embedding_function=embedding_function)

        token_classless_few = persistent_client_classless_few.get_collection(token_classless_few_name, embedding_function=embedding_function)

        # test to make sure the collections are working as intended
        test_metrics = CollectionStats("test_2")
        test_metrics.get_stats(token_class_few_full)
        temp1 = f"Token_few_full:\n\nmetrics.stdev: {test_metrics.stdev:.4f}\n\nmetrics.agg_stdev: {test_metrics.agg_stdev:.4f}\n\nmetrics.avg_stdev: {test_metrics.avg_stdev:.4f}\n\nmetrics.bimod: {test_metrics.bimod:.4f}\n\nmetrics.skewness: {test_metrics.skewness:.4f}\n\nmetrics.kurtosis: {test_metrics.kurtosis:.4f}"
        test_metrics.get_stats(token_class_few_half)
        temp2 = f"Token_few_half:\n\nmetrics.stdev: {test_metrics.stdev:.4f}\n\nmetrics.agg_stdev: {test_metrics.agg_stdev:.4f}\n\nmetrics.avg_stdev: {test_metrics.avg_stdev:.4f}\n\nmetrics.bimod: {test_metrics.bimod:.4f}\n\nmetrics.skewness: {test_metrics.skewness:.4f}\n\nmetrics.kurtosis: {test_metrics.kurtosis:.4f}"
        test_metrics.get_stats(token_class_few_quarter)
        temp3 = f"Token_few_quarter:\n\nmetrics.stdev: {test_metrics.stdev:.4f}\n\nmetrics.agg_stdev: {test_metrics.agg_stdev:.4f}\n\nmetrics.avg_stdev: {test_metrics.avg_stdev:.4f}\n\nmetrics.bimod: {test_metrics.bimod:.4f}\n\nmetrics.skewness: {test_metrics.skewness:.4f}\n\nmetrics.kurtosis: {test_metrics.kurtosis:.4f}"
        test_metrics.get_stats(token_class_zero)
        temp4 =  f"Token_zero:\n\nmetrics.stdev: {test_metrics.stdev:.4f}\n\nmetrics.agg_stdev: {test_metrics.agg_stdev:.4f}\n\nmetrics.avg_stdev: {test_metrics.avg_stdev:.4f}\n\nmetrics.bimod: {test_metrics.bimod:.4f}\n\nmetrics.skewness: {test_metrics.skewness:.4f}\n\nmetrics.kurtosis: {test_metrics.kurtosis:.4f}"
        test_metrics.get_stats(token_classless_few)
        temp5 =  f"Token_few_classless:\n\nmetrics.stdev: {test_metrics.stdev:.4f}\n\nmetrics.agg_stdev: {test_metrics.agg_stdev:.4f}\n\nmetrics.avg_stdev: {test_metrics.avg_stdev:.4f}\n\nmetrics.bimod: {test_metrics.bimod:.4f}\n\nmetrics.skewness: {test_metrics.skewness:.4f}\n\nmetrics.kurtosis: {test_metrics.kurtosis:.4f}"

        #print("There are", token_class_few_full.count(), "in the collection")
        #print("There are", token_class_few_half.count(), "in the collection")
        #print("There are", token_class_few_quarter.count(), "in the collection")
        #print("There are", token_class_zero.count(), "in the collection")
        #print("There are", token_classless_few.count(), "in the collection")

        print("------------------------------")
        print(temp1)
        print("------------------------------")
        print(temp2)
        print("------------------------------")
        print(temp3)
        print("------------------------------")
        print(temp4)
        print("------------------------------")
        print(temp5)
        print("------------------------------")

        if TOKEN_MODELS:
            # encoded dataset locations
            class_few_full_path = "./dataset/encoded_tokens/class_few_full.pth"
            class_few_half_path = "./dataset/encoded_tokens/class_few_half.pth"
            class_few_quarter_path = "./dataset/encoded_tokens/class_few_quarter.pth"
            class_zero_path = "./dataset/encoded_tokens/class_zero.pth"
            classless_few_path = "./dataset/encoded_tokens/classless_few.pth"

            if os.path.exists(class_few_full_path):
                token_class_few_full_dataset = torch.load(class_few_full_path)
            else:
                token_class_few_full_dataset = preprocess_tokenized_dataset_collection(token_class_few_full, checkpoint)
                torch.save(token_class_few_full_dataset, class_few_full_path)
            if os.path.exists(class_few_half_path):
                token_class_few_half_dataset = torch.load(class_few_half_path)
            else:
                token_class_few_half_dataset = preprocess_tokenized_dataset_collection(token_class_few_half, checkpoint)
                torch.save(token_class_few_half_dataset, class_few_half_path)
            if os.path.exists(class_few_quarter_path):
                token_class_few_quarter_dataset = torch.load(class_few_quarter_path)
            else:
                token_class_few_quarter_dataset = preprocess_tokenized_dataset_collection(token_class_few_quarter, checkpoint)
                torch.save(token_class_few_quarter_dataset, class_few_quarter_path)
            if os.path.exists(class_zero_path):
                token_class_zero_dataset = torch.load(class_zero_path)
            else:
                token_class_zero_dataset = preprocess_tokenized_dataset_collection(token_class_zero, checkpoint)
                torch.save(token_class_zero_dataset, class_zero_path)
            if os.path.exists(classless_few_path):
                token_classless_few_dataset = torch.load(classless_few_path)
            else:
                token_classless_few_dataset = preprocess_tokenized_dataset_collection(token_classless_few, checkpoint)
                torch.save(token_classless_few_dataset, classless_few_path)

            load_seed_dataset_tree = torch.load('./dataset/encoded_tokens/seed_tokens.pth')

            # configuration values here:
            LR_RATE = 3e-4
            BATCH_SIZE = 32
            MAX_EPOCHS = 2
            DEVICE = "cuda:1"
            NUM_RUNS = 3

            # model values here
            TOK_EMBED_DIM = 256
            TOK_IN_CHANNELS = 72
            TOK_HIDDEN_LAY1 = 36
            TOK_HIDDEN_LAY2 = 18
            GRAPH_CLASS = 1

            # now to get all the datasets into datalaoders
            loader_seed_dataset_tree = DataLoader(dataset=load_seed_dataset_tree, batch_size=BATCH_SIZE, shuffle=True)
            loader_tree_class_few_full_dataset = DataLoader(dataset=token_class_few_full_dataset, batch_size=BATCH_SIZE, shuffle=True)
            loader_tree_class_few_half_dataset = DataLoader(dataset=token_class_few_half_dataset, batch_size=BATCH_SIZE, shuffle=True)
            loader_tree_class_few_quarter_dataset = DataLoader(dataset=token_class_few_quarter_dataset, batch_size=BATCH_SIZE, shuffle=True)
            loader_tree_class_zero_dataset = DataLoader(dataset=token_class_zero_dataset, batch_size=BATCH_SIZE, shuffle=True)
            loader_tree_classless_few_dataset = DataLoader(dataset=token_classless_few_dataset, batch_size=BATCH_SIZE, shuffle=True)

            # to make it easier to iterate over all the dataset, we will add them into a list now, with a name for the second entry in the tuple
            loader_list = []
            loader_list.append((loader_seed_dataset_tree, "seed_token"))
            loader_list.append((loader_tree_class_few_full_dataset, "class_few_full_token"))
            loader_list.append((loader_tree_class_few_half_dataset, "class_few_half_token"))
            loader_list.append((loader_tree_class_few_quarter_dataset, "class_few_quarter_token"))
            loader_list.append((loader_tree_class_zero_dataset, "class_zero_token"))
            loader_list.append((loader_tree_classless_few_dataset, "classless_few_token"))

            # now to setup a model, train it, get the metrics, save the model, and then reset everything for the next dataset
            for run in range(NUM_RUNS):
                for (loader, name) in loader_list:
                    model = TokenClassificationModel2(TOK_EMBED_DIM, TOK_IN_CHANNELS, TOK_HIDDEN_LAY1, TOK_HIDDEN_LAY2, GRAPH_CLASS).to(DEVICE)
                    optimizer = optim.Adam(model.parameters(), lr=LR_RATE) # adam optimizer
                    loss_function = nn.BCELoss()
                    model.train()

                    for epoch in range(1, MAX_EPOCHS):
                        overall_loss = 0
                        for batch_idx, (data, label) in enumerate(loader):
                            data = data.to(DEVICE)
                            label = label.to(DEVICE)
                            optimizer.zero_grad()
                            output = model(data)
                            loss = loss_function(output.view(label.shape[0]), label.float())
                            overall_loss += loss.item()
                            loss.backward()
                            optimizer.step()
                        print("\tSeed Epoch", epoch, "\tSeed Average Loss: ", overall_loss/((batch_idx+1)*BATCH_SIZE))

                    # now to save the model 
                    torch.save(model.state_dict(), f"./models/token_models/{name}/model_{run}.pth")
                    del model


    if TREE_METRICS:
        tree_class_few_full_name = "tree_class_few_full3"
        tree_class_few_half_name = "tree_class_few_half3"
        tree_class_few_quarter_name = "tree_class_few_quarter3"
        tree_class_zero_name = "tree_class_zero3"
        tree_classless_few_name = "tree_classless_few3"

        checkpoint = "Salesforce/codet5p-110m-embedding"
        tree_checkpoint = "./models/tree_ae/tree_ae.pth"
        url = "http://localhost:8000/code2fn/fn-given-filepaths"
        embedding_function = ChromaTreeEmbedding(checkpoint, tree_checkpoint, url)

        # spool up 5 clients
        persistent_client_class_few_full = chromadb.PersistentClient()
        persistent_client_class_few_half = chromadb.PersistentClient()
        persistent_client_class_few_quarter = chromadb.PersistentClient()
        persistent_client_class_zero = chromadb.PersistentClient()
        persistent_client_classless_few = chromadb.PersistentClient()

        # grab the collections
        tree_class_few_full = persistent_client_class_few_full.get_collection(tree_class_few_full_name, embedding_function=embedding_function)

        tree_class_few_half = persistent_client_class_few_half.get_collection(tree_class_few_half_name, embedding_function=embedding_function)

        tree_class_few_quarter = persistent_client_class_few_quarter.get_collection(tree_class_few_quarter_name, embedding_function=embedding_function)

        tree_class_zero = persistent_client_class_zero.get_collection(tree_class_zero_name, embedding_function=embedding_function)

        tree_classless_few = persistent_client_classless_few.get_collection(tree_classless_few_name, embedding_function=embedding_function)

        #print("There are", tree_class_few_full.count(), "in the collection")
        #print("There are", tree_class_few_half.count(), "in the collection")
        #print("There are", tree_class_few_quarter.count(), "in the collection")
        #print("There are", tree_class_zero.count(), "in the collection")
        #print("There are", tree_classless_few.count(), "in the collection")

        # test to make sure the collections are working as intended
        test_metrics = CollectionStats("test_2")
        test_metrics.get_stats(tree_class_few_full)
        temp1 = f"Tree_few_full:\n\nmetrics.stdev: {test_metrics.stdev:.4f}\n\nmetrics.agg_stdev: {test_metrics.agg_stdev:.4f}\n\nmetrics.avg_stdev: {test_metrics.avg_stdev:.4f}\n\nmetrics.bimod: {test_metrics.bimod:.4f}\n\nmetrics.skewness: {test_metrics.skewness:.4f}\n\nmetrics.kurtosis: {test_metrics.kurtosis:.4f}"
        test_metrics.get_stats(tree_class_few_half)
        temp2 = f"Tree_few_half:\n\nmetrics.stdev: {test_metrics.stdev:.4f}\n\nmetrics.agg_stdev: {test_metrics.agg_stdev:.4f}\n\nmetrics.avg_stdev: {test_metrics.avg_stdev:.4f}\n\nmetrics.bimod: {test_metrics.bimod:.4f}\n\nmetrics.skewness: {test_metrics.skewness:.4f}\n\nmetrics.kurtosis: {test_metrics.kurtosis:.4f}"
        test_metrics.get_stats(tree_class_few_quarter)
        temp3 = f"Tree_few_quarter:\n\nmetrics.stdev: {test_metrics.stdev:.4f}\n\nmetrics.agg_stdev: {test_metrics.agg_stdev:.4f}\n\nmetrics.avg_stdev: {test_metrics.avg_stdev:.4f}\n\nmetrics.bimod: {test_metrics.bimod:.4f}\n\nmetrics.skewness: {test_metrics.skewness:.4f}\n\nmetrics.kurtosis: {test_metrics.kurtosis:.4f}"
        test_metrics.get_stats(tree_class_zero)
        temp4 = f"Tree_zero:\n\nmetrics.stdev: {test_metrics.stdev:.4f}\n\nmetrics.agg_stdev: {test_metrics.agg_stdev:.4f}\n\nmetrics.avg_stdev: {test_metrics.avg_stdev:.4f}\n\nmetrics.bimod: {test_metrics.bimod:.4f}\n\nmetrics.skewness: {test_metrics.skewness:.4f}\n\nmetrics.kurtosis: {test_metrics.kurtosis:.4f}"
        test_metrics.get_stats(tree_classless_few)
        temp5 = f"Tree_few_classless:\n\nmetrics.stdev: {test_metrics.stdev:.4f}\n\nmetrics.agg_stdev: {test_metrics.agg_stdev:.4f}\n\nmetrics.avg_stdev: {test_metrics.avg_stdev:.4f}\n\nmetrics.bimod: {test_metrics.bimod:.4f}\n\nmetrics.skewness: {test_metrics.skewness:.4f}\n\nmetrics.kurtosis: {test_metrics.kurtosis:.4f}"

        print("------------------------------")
        print(temp1)
        print("------------------------------")
        print(temp2)
        print("------------------------------")
        print(temp3)
        print("------------------------------")
        print(temp4)
        print("------------------------------")
        print(temp5)
        print("------------------------------")
        if TREE_MODELS:
            # see if the dataset have already been generated
            class_few_full_path = "./dataset/encoded_trees/class_few_full3.pth"
            class_few_half_path = "./dataset/encoded_trees/class_few_half3.pth"
            class_few_quarter_path = "./dataset/encoded_trees/class_few_quarter3.pth"
            class_zero_path = "./dataset/encoded_trees/class_zero3.pth"
            classless_few_path = "./dataset/encoded_trees/classless_few3.pth"

            if os.path.exists(class_few_full_path):
                tree_class_few_full_dataset = torch.load(class_few_full_path)
            else:
                tree_class_few_full_dataset = clean_tree_dataset(preprocess_tree_collection_dataset(tree_class_few_full, url, checkpoint))
                torch.save(tree_class_few_full_dataset, class_few_full_path)
            if os.path.exists(class_few_half_path):
                tree_class_few_half_dataset = torch.load(class_few_half_path)
            else:
                tree_class_few_half_dataset = clean_tree_dataset(preprocess_tree_collection_dataset(tree_class_few_half, url, checkpoint))
                torch.save(tree_class_few_half_dataset, class_few_half_path)
            if os.path.exists(class_few_quarter_path):
                tree_class_few_quarter_dataset = torch.load(class_few_quarter_path)
            else:
                tree_class_few_quarter_dataset = clean_tree_dataset(preprocess_tree_collection_dataset(tree_class_few_quarter, url, checkpoint))
                torch.save(tree_class_few_quarter_dataset, class_few_quarter_path)
            if os.path.exists(class_zero_path):
                tree_class_zero_dataset = torch.load(class_zero_path)
            else:
                tree_class_zero_dataset = clean_tree_dataset(preprocess_tree_collection_dataset(tree_class_zero, url, checkpoint))
                torch.save(tree_class_zero_dataset, class_zero_path)
            if os.path.exists(classless_few_path):
                tree_classless_few_dataset = torch.load(classless_few_path)
            else:
                tree_classless_few_dataset = clean_tree_dataset(preprocess_tree_collection_dataset(tree_classless_few, url, checkpoint))
                torch.save(tree_classless_few_dataset, classless_few_path)

            load_seed_dataset_tree = clean_tree_dataset(torch.load('./dataset/encoded_trees/seed_trees.pth'))

            # configuration values here:
            LR_RATE = 3e-4
            BATCH_SIZE = 32
            MAX_EPOCHS = 6
            GRAPH_CLASS = 1
            DEVICE = "cuda:1"
            NUM_RUNS = 3

            # now to get all the datasets into datalaoders
            loader_seed_dataset_tree = GeometricLoader(dataset=load_seed_dataset_tree, batch_size=BATCH_SIZE, shuffle=True)
            loader_tree_class_few_full_dataset = GeometricLoader(dataset=tree_class_few_full_dataset, batch_size=BATCH_SIZE, shuffle=True)
            loader_tree_class_few_half_dataset = GeometricLoader(dataset=tree_class_few_half_dataset, batch_size=BATCH_SIZE, shuffle=True)
            loader_tree_class_few_quarter_dataset = GeometricLoader(dataset=tree_class_few_quarter_dataset, batch_size=BATCH_SIZE, shuffle=True)
            loader_tree_class_zero_dataset = GeometricLoader(dataset=tree_class_zero_dataset, batch_size=BATCH_SIZE, shuffle=True)
            loader_tree_classless_few_dataset = GeometricLoader(dataset=tree_classless_few_dataset, batch_size=BATCH_SIZE, shuffle=True)

            # to make it easier to iterate over all the dataset, we will add them into a list now, with a name for the second entry in the tuple
            loader_list = []
            loader_list.append((loader_seed_dataset_tree, "seed_tree"))
            loader_list.append((loader_tree_class_few_full_dataset, "class_few_full_tree"))
            loader_list.append((loader_tree_class_few_half_dataset, "class_few_half_tree"))
            loader_list.append((loader_tree_class_few_quarter_dataset, "class_few_quarter_tree"))
            loader_list.append((loader_tree_class_zero_dataset, "class_zero_tree"))
            loader_list.append((loader_tree_classless_few_dataset, "classless_few_tree"))

            # now to setup a model, train it, get the metrics, save the model, and then reset everything for the next dataset
            for run in range(NUM_RUNS):
                for (loader, name) in loader_list:
                    model = TreeClassificationModel(EMBED_DIM, IN_CHANNELS, HIDDEN_CHANNELS, OUT_CHANNELS, NODE_CLASSES, COMPRESSED_CLASSES, GRAPH_CLASS).to(DEVICE)
                    optimizer = optim.Adam(model.parameters(), lr=LR_RATE) # adam optimizer
                    loss_function = nn.BCELoss()
                    model.train()

                    for epoch in range(1, MAX_EPOCHS):
                        overall_loss = 0
                        for batch_idx, (data, label) in enumerate(loader):
                            # need to use data.shape[0] as batch size in view incase dataset is not evenly divisble by 32
                            data = data.to(DEVICE)
                            label = label.to(DEVICE)
                            optimizer.zero_grad()
                            output = model(data.x1, data.x2, data.pe, data.edge_index, data.batch)
                            loss = loss_function(output.view(label.shape[0]), label.float())
                            overall_loss += loss.item()
                            loss.backward()
                            optimizer.step()
                        print("\tSeed Epoch", epoch + 1, "\tSeed Average Loss: ", overall_loss/((batch_idx+1)*BATCH_SIZE))

                    # now to save the model 
                    torch.save(model.state_dict(), f"./models/tree_models/{name}/model_{run}.pth")
                    del model

                