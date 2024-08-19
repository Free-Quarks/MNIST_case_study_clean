# langchain version 0.2, 07/03/2024
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import CharacterTextSplitter
import chromadb
from chromadb import Documents, EmbeddingFunction

import os
from transformers import AutoModel, AutoTokenizer
from typing import List
import random as rd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


from tree_encoder_model import GNNModel
from code_preprocessing import preprocess_tree_query
import torch

from vector_db import ChromaCodet5pEmbedding, ChromaTreeEmbedding

from tree_model_trainer import EMBED_DIM, IN_CHANNELS, HIDDEN_CHANNELS, OUT_CHANNELS, NODE_CLASSES, COMPRESSED_CLASSES, COMPRESSED_GRAPH_FEATURE, GRAPH_FEATURE

DEVICE = "cuda"
CHECKPOINT = "Salesforce/codet5p-110m-embedding"
TREE_CHECKPOINT = "./models/tree_ae/tree_ae.pth"
URL = "http://localhost:8000/code2fn/fn-given-filepaths"

def collection_2_class_stats(collection):
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

    mean_0 = np.mean(class_0_array)
    mean_1 = np.mean(class_1_array)
    stdev_0 = np.std(class_0_array, axis=0)
    stdev_1 = np.std(class_1_array, axis=0)
    stdev_mean_0 = np.mean(stdev_0)
    stdev_mean_1 = np.mean(stdev_1)
    
    class_stats = {
        'class': [
                {
                'mean': mean_0,
                'stdev': stdev_0,
                'stdev_mean': stdev_mean_0
            }, 
                {
                'mean': mean_1,
                'stdev': stdev_1,
                'stdev_mean': stdev_mean_1
            }
        ]
    }

    return class_stats, (class_0_array, class_1_array)





def collections_2_clustering_plots(seed_collection, trim_collection, diverse_collection, figure_save_location):
    tree_seed_embed_meta = seed_collection.get(include=["embeddings", "metadatas"])
    tree_trim_embed_meta = trim_collection.get(include=["embeddings", "metadatas"])
    tree_diverse_embed_meta = diverse_collection.get(include=["embeddings", "metadatas"])

    seed_data, seed_labels = vecdb_2_labeled_embeddings(tree_seed_embed_meta)
    trim_data, trim_labels = vecdb_2_labeled_embeddings(tree_trim_embed_meta)
    diverse_data, diverse_labels = vecdb_2_labeled_embeddings(tree_diverse_embed_meta)
    combined_data = np.concatenate((trim_data, diverse_data))
    combined_labels = np.concatenate((trim_labels, diverse_labels))
    
    tsne = TSNE(n_components=2, random_state=42)
    tsne_results_seed = tsne.fit_transform(seed_data)
    tsne_results_trim = tsne.fit_transform(trim_data)
    tsne_results_diverse = tsne.fit_transform(diverse_data)
    tsne_results_combined = tsne.fit_transform(combined_data)

    plt.figure(figsize=(12, 10))

    plt.subplot(2, 2, 1)
    for label in np.unique(seed_labels):
        plt.scatter(tsne_results_seed[seed_labels == label, 0], tsne_results_seed[seed_labels == label, 1], alpha=0.7)
    plt.title('Seed')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.legend(["compartmental", "abm"])
    plt.grid(True)

    plt.subplot(2, 2, 2)
    for label in np.unique(trim_labels):
        plt.scatter(tsne_results_trim[trim_labels == label, 0], tsne_results_trim[trim_labels == label, 1], alpha=0.7)
    plt.title('Trimmed')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.legend(["compartmental", "abm"])
    plt.grid(True)

    plt.subplot(2, 2, 3)
    for label in np.unique(diverse_labels):
        plt.scatter(tsne_results_diverse[diverse_labels == label, 0], tsne_results_diverse[diverse_labels == label, 1], alpha=0.7)
    plt.title('Diverse')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.legend(["compartmental", "abm"])
    plt.grid(True)

    plt.subplot(2, 2, 4)
    for label in np.unique(combined_labels):
        plt.scatter(tsne_results_combined[combined_labels == label, 0], tsne_results_combined[combined_labels == label, 1], alpha=0.7)
    plt.title('Combined')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.legend(["compartmental", "abm"])
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(figure_save_location)


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


def array_2_mean_stdev(vec):
    array = np.array(vec) 
    stdev_vec = np.std(array, axis=0)
    mean_stdev = np.mean(stdev_vec)
    return mean_stdev

# returns a vector of indexes to delete from a dataset due to being too similar to another data point in the dataset
# note: the vector is return in reverse order, so the largest vector is first, to allow easy deletion from the orignal
# dataset. 
def idxs_2_trim(vec_of_embeddings, num_idxs):
    marginalized_distances = []
    for (i, ent1) in enumerate(vec_of_embeddings):
        distance_margin = []
        for (j, ent2) in enumerate(vec_of_embeddings):
            if j > i:
                distance = np.linalg.norm(np.array(ent1) - np.array(ent2))
                distance_margin.append(distance)
        if len(distance_margin) > 0:
            min_dist = np.min(np.array(distance_margin))
            marg_dist_tup = (i, min_dist)
            marginalized_distances.append(marg_dist_tup)

    sorted_distances = sorted(marginalized_distances, key=lambda x: x[1])

    idxs_unsorted = []
    
    for (index, _dist) in sorted_distances[:num_idxs]:
        idxs_unsorted.append(index)

    idxs = sorted(idxs_unsorted, reverse=True)

    return idxs

if __name__ == "__main__":
    # seed token vector db #--------------------
    embedding_function_chroma_codet = ChromaCodet5pEmbedding(CHECKPOINT)
    persistent_client_tok = chromadb.PersistentClient() # default settings
    # this gets the collection since it's already present
    collection_seed_tokens = persistent_client_tok.get_collection("seed_code", embedding_function=embedding_function_chroma_codet)
    print("There are", collection_seed_tokens.count(), "in the collection")
    collection_seed_token_vecs = collection_seed_tokens.get(include=["embeddings"])


    # seed tree vector db #--------------------
    embedding_function_chroma_tree = ChromaTreeEmbedding(CHECKPOINT, TREE_CHECKPOINT, URL)
    persistent_client_tree = chromadb.PersistentClient() # default settings
    # this gets the collection since it's already present
    collection_seed_tree = persistent_client_tree.get_collection("seed_code_tree", embedding_function=embedding_function_chroma_tree)
    collection_seed_tree_vecs = collection_seed_tree.get(include=["embeddings"])

    #--------------------
    # need to construct diversified vector db's
    # setup import directories
    seed_data_dir = "./dataset/new_test_code"
    tok_diverse_data_dir = "./dataset/agentic_data_token_v2"
    tree_diverse_data_dir = "./dataset/agentic_data_tree_v2"

    raw_docs_seed = DirectoryLoader(seed_data_dir, glob='*.py', loader_cls=TextLoader).load()
    #--------------------
    persistent_client_tok_temp_seed = chromadb.PersistentClient() # default settings
    # this gets the collection since it's already present
    collection_seed_tokens_temp = persistent_client_tok_temp_seed.get_or_create_collection("seed_code_tok_temp", embedding_function=embedding_function_chroma_codet)

    if collection_seed_tokens_temp.count() < 10:
        for i, entry in enumerate(raw_docs_seed):
            collection_seed_tokens_temp.add(ids=f"{i}", embeddings = embedding_function_chroma_codet(entry.page_content), metadatas=entry.metadata, documents=entry.page_content)
            print(f"{i} of {len(raw_docs_seed)} added to db")

    collection_seed_tree_vecs_temp = collection_seed_tokens_temp.get(include=["embeddings"])
    # token diversified vector db #--------------------
    # grab the docs
    text_splitter = CharacterTextSplitter(chunk_size=456, chunk_overlap=0)

    raw_docs_tok_diverse = DirectoryLoader(tok_diverse_data_dir, glob='*.py', loader_cls=TextLoader).load()
    
    # instaniate a temp client
    temp_client_tok = chromadb.PersistentClient() # in memory only since only used for analytics
    collection_diverse_tok = temp_client_tok.get_or_create_collection("diverse_code_tok", embedding_function=embedding_function_chroma_codet)

    temp_client_trim_tok = chromadb.PersistentClient() # in memory only since only used for analytics
    collection_trim_tok = temp_client_trim_tok.get_or_create_collection("trim_code_tok_v2", embedding_function=embedding_function_chroma_codet)

    # merge diverse and seed data
    data_to_del = idxs_2_trim(collection_seed_tree_vecs_temp['embeddings'], len(raw_docs_tok_diverse))

    # del subset of data, replace with new diverse data
    raw_docs_seed_trim = raw_docs_seed

    print(data_to_del[0])
    print(len(raw_docs_seed_trim))

    for idx in data_to_del:
        raw_docs_seed_trim.pop(idx)

    docs_tok_diverse = text_splitter.split_documents(raw_docs_tok_diverse)
    docs_tok_trim = text_splitter.split_documents(raw_docs_seed_trim)
    # create the db
    if collection_diverse_tok.count() < 10:
        for i, entry in enumerate(docs_tok_diverse):
            collection_diverse_tok.add(ids=f"{i}", embeddings = embedding_function_chroma_codet(entry.page_content), metadatas=entry.metadata, documents=entry.page_content)
            print(f"{i} of {len(docs_tok_diverse)} added to db")
    if collection_trim_tok.count() < 10:
        for i, entry in enumerate(docs_tok_trim):
            collection_trim_tok.add(ids=f"{i}", embeddings = embedding_function_chroma_codet(entry.page_content), metadatas=entry.metadata, documents=entry.page_content)
            print(f"{i} of {len(docs_tok_trim)} added to db")

    # pull all the vectors
    collection_trim_tok_vecs = collection_trim_tok.get(include=["embeddings"])
    collection_diverse_tok_vecs = collection_diverse_tok.get(include=["embeddings"])

    # tree diversfied vector db #--------------------
    # grab the docs
    raw_docs_tree_diverse = DirectoryLoader(tree_diverse_data_dir, glob='*.py', loader_cls=TextLoader).load()

    # instaniate temp client
    temp_client_tree = chromadb.PersistentClient() # in memory only since only used for analytics
    collection_diverse_tree = temp_client_tree.get_or_create_collection("diverse_code_tree_v2", embedding_function=embedding_function_chroma_tree)

    temp_client_trim_tree = chromadb.PersistentClient() # in memory only since only used for analytics
    collection_trim_tree = temp_client_trim_tree.get_or_create_collection("trim_code_tree_v2", embedding_function=embedding_function_chroma_tree)

    # merge diverse and seed data
    data_to_del = idxs_2_trim(collection_seed_tree_vecs['embeddings'], len(raw_docs_tree_diverse))

    # del subset of data, replace with new diverse data
    raw_docs_seed_trim_tree = raw_docs_seed

    for idx in data_to_del:
        raw_docs_seed_trim_tree.pop(idx)

    # create the db
    if collection_diverse_tree.count() < 10:
        for i, entry in enumerate(raw_docs_tree_diverse):
            collection_diverse_tree.add(ids=f"{i}", embeddings = embedding_function_chroma_tree(entry.page_content), metadatas=entry.metadata, documents=entry.page_content)
            print(f"{i} of {len(raw_docs_tree_diverse)} added to db")
    if collection_trim_tree.count() < 10:
        for i, entry in enumerate(raw_docs_seed_trim_tree):
            collection_trim_tree.add(ids=f"{i}", embeddings = embedding_function_chroma_tree(entry.page_content), metadatas=entry.metadata, documents=entry.page_content)
            print(f"{i} of {len(raw_docs_seed_trim_tree)} added to db")

    # pull all the vectors
    collection_diverse_tree_vecs = collection_diverse_tree.get(include=["embeddings"])
    collection_trim_tree_vecs = collection_trim_tree.get(include=["embeddings"])

    # for token diverse dataset under tree metrics #--------------------
    temp_client_tok_tree = chromadb.PersistentClient() # in memory only since only used for analytics
    collection_diverse_tok_tree = temp_client_tok_tree.get_or_create_collection("diverse_code_tok_tree", embedding_function=embedding_function_chroma_tree)

    if collection_diverse_tok_tree.count() < 10:
        for i, entry in enumerate(raw_docs_tok_diverse):
            collection_diverse_tok_tree.add(ids=f"{i}", embeddings = embedding_function_chroma_tree(entry.page_content), metadatas=entry.metadata, documents=entry.page_content)
            print(f"{i} of {len(raw_docs_tok_diverse)} added to db")

    collection_diverse_tok_tree_vecs = collection_diverse_tok_tree.get(include=["embeddings"])

    # for the tree diverse dataset under the token metrics

    temp_client_tree_tok = chromadb.PersistentClient() # in memory only since only used for analytics
    collection_diverse_tree_tok = temp_client_tree_tok.get_or_create_collection("diverse_code_tree_tok_v2", embedding_function=embedding_function_chroma_tree)

    docs_tree_tok_diverse = text_splitter.split_documents(raw_docs_tree_diverse)
    if collection_diverse_tree_tok.count() < 10:
        for i, entry in enumerate(docs_tree_tok_diverse):
            collection_diverse_tree_tok.add(ids=f"{i}", embeddings = embedding_function_chroma_codet(entry.page_content), metadatas=entry.metadata, documents=entry.page_content)
            print(f"{i} of {len(docs_tree_tok_diverse)} added to db")

    collection_diverse_tree_tok_vecs = collection_diverse_tree_tok.get(include=["embeddings"])

    """
    we now have several vectors for metrics and analytics:

        collection_seed_token_vecs: seed dataset under token metrics

        collection_seed_tree_vecs: seed dataset under tree metrics

        collection_diverse_tok_vecs: token diverse data only, under token metric

        collection_trim_tok_vecs: trimmed seed data only, under token metric
        
        collection_diverse_tree_vecs: tree diverse data only, under tree metric

        collection_trim_tree_vecs: trimmed seed data only, under tree metric

        collection_diverse_tok_tree_vecs: token diverse data only, under tree metric

        collection_diverse_tree_tok_vecs: tree diverse data only, under token metric
    
    Below we now compute the analytics of the vectors
    """
    
    diverse_tok_tok = collection_trim_tok_vecs['embeddings'] + collection_diverse_tok_vecs['embeddings']
    diverse_tree_tree = collection_trim_tree_vecs['embeddings'] + collection_diverse_tree_vecs['embeddings']
    diverse_tok_tree = collection_trim_tree_vecs['embeddings'] + collection_diverse_tok_tree_vecs['embeddings']
    diverse_tree_tok = collection_trim_tok_vecs['embeddings'] + collection_diverse_tree_tok_vecs['embeddings']

    expanded_diverse_token = collection_seed_token_vecs['embeddings'] + collection_diverse_tok_vecs['embeddings']
    expanded_diverse_tree = collection_seed_tree_vecs['embeddings'] + collection_diverse_tree_vecs['embeddings']

    print(f"token diverse len: {len(collection_diverse_tok_vecs['embeddings']):.4f}")
    print(f"tree diverse len: {len(collection_diverse_tree_vecs['embeddings']):.4f}")

    print(f"seed token: {array_2_mean_stdev(collection_seed_token_vecs['embeddings']):.4f}")
    print(f"seed tree: {array_2_mean_stdev(collection_seed_tree_vecs['embeddings']):.4f}")
    print("----------------------------------------------------")
    print(f"trim token: {array_2_mean_stdev(collection_trim_tok_vecs['embeddings']):.4f}")
    print(f"trim tree: {array_2_mean_stdev(collection_trim_tree_vecs['embeddings']):.4f}")
    print("----------------------------------------------------")
    print(f"expanded diverse token: {array_2_mean_stdev(expanded_diverse_token):.4f}")
    print(f"expanded diverse tree: {array_2_mean_stdev(expanded_diverse_tree):.4f}")
    print("----------------------------------------------------")
    print(f"diverse token: {array_2_mean_stdev(diverse_tok_tok):.4f}")
    print(f"diverse tree: {array_2_mean_stdev(diverse_tree_tree):.4f}")
    print("----------------------------------------------------")
    print(f"diverse tree in token: {array_2_mean_stdev(diverse_tree_tok):.4f}")
    print(f"diverse token in tree: {array_2_mean_stdev(diverse_tok_tree):.4f}")

    # now let's look at clustering some of the data as well. Let's cluster tree_seed and tree_diverse
    #collections_2_clustering_plots(collection_seed_tree, collection_trim_tree, collection_diverse_tree, "./figs/tree-tsne-results.png")
    #collections_2_clustering_plots(collection_seed_tokens, collection_trim_tok, collection_diverse_tok, "./figs/token-tsne-results.png")

    class_stats_seed_tree, seed_arrays = collection_2_class_stats(collection_seed_tree)
    class_stats_trim_tree, trim_arrays = collection_2_class_stats(collection_trim_tree)
    diverse_0 = np.concatenate((seed_arrays[0], trim_arrays[0]))
    diverse_1 = np.concatenate((seed_arrays[1], trim_arrays[1]))
    
    diverse_0_stdev = np.std(diverse_0, axis=0)
    diverse_0_stdev_mean = np.mean(diverse_0_stdev)
    diverse_1_stdev = np.std(diverse_1, axis=0)
    diverse_1_stdev_mean = np.mean(diverse_1_stdev)

    print("----------------------------------------------------")
    print("----------------------------------------------------")
    #print(f"seed mean 0: {class_stats_seed_tree['class'][0]['mean']}, seed mean 1: {class_stats_seed_tree['class'][1]['mean']}")
    print(f"tree seed stdev 0: {class_stats_seed_tree['class'][0]['stdev_mean']:.4f}, tree seed stdev 1: {class_stats_seed_tree['class'][1]['stdev_mean']:.4f}")
    #print(f"trim mean 0: {class_stats_trim_tree['class'][0]['mean']}, trim mean 1: {class_stats_trim_tree['class'][1]['mean']}")
    print(f"tree trim stdev 0: {class_stats_trim_tree['class'][0]['stdev_mean']:.4f}, tree trim stdev 1: {class_stats_trim_tree['class'][1]['stdev_mean']:.4f}")
    print(f"tree dive stdev 0: {diverse_0_stdev_mean:.4f}, tree dive stdev 1: {diverse_1_stdev_mean:.4f}")
    print("----------------------------------------------------")
    print("Tree:")
    print(f"\tClass 0 trimmed: {100*(1-(len(trim_arrays[0])/len(seed_arrays[0]))):.2f}%")
    print(f"\tClass 1 trimmed: {100*(1-(len(trim_arrays[1])/len(seed_arrays[1]))):.2f}%")
    print("----------------------------------------------------")
    class_stats_seed_tree, seed_arrays = collection_2_class_stats(collection_seed_tokens)
    class_stats_trim_tree, trim_arrays = collection_2_class_stats(collection_trim_tok)
    diverse_0 = np.concatenate((seed_arrays[0], trim_arrays[0]))
    diverse_1 = np.concatenate((seed_arrays[1], trim_arrays[1]))
    
    diverse_0_stdev = np.std(diverse_0, axis=0)
    diverse_0_stdev_mean = np.mean(diverse_0_stdev)
    diverse_1_stdev = np.std(diverse_1, axis=0)
    diverse_1_stdev_mean = np.mean(diverse_1_stdev)
    print(f"token seed stdev 0: {class_stats_seed_tree['class'][0]['stdev_mean']:.4f}, token seed stdev 1: {class_stats_seed_tree['class'][1]['stdev_mean']:.4f}")
    print(f"token trim stdev 0: {class_stats_trim_tree['class'][0]['stdev_mean']:.4f}, token trim stdev 1: {class_stats_trim_tree['class'][1]['stdev_mean']:.4f}")
    print(f"token dive stdev 0: {diverse_0_stdev_mean:.4f}, token dive stdev 1: {diverse_1_stdev_mean:.4f}")

    print("----------------------------------------------------")
    print("Token:")
    print(f"\tClass 0 trimmed: {100*(1-(len(trim_arrays[0])/len(seed_arrays[0]))):.2f}%")
    print(f"\tClass 1 trimmed: {100*(1-(len(trim_arrays[1])/len(seed_arrays[1]))):.2f}%")
