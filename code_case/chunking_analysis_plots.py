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
from vector_db_analytics import vecdb_2_labeled_embeddings

DEVICE = "cuda"
CHECKPOINT = "Salesforce/codet5p-110m-embedding"

def chunked_vecdb_2_aggregated_labeled_embeddings(collection):
    collection_dict = collection.get(include=["embeddings", "metadatas"])
    metadata_list = collection_dict['metadatas']
    source_list = []
    for metadata in metadata_list:
        if metadata['source'] not in source_list:
            source_list.append(metadata['source'])
    
    output_data = []
    output_labels = []
    for source in source_list:
        label = 0
        if source.split("-")[-2] == "abm":
            label = 1
        ents = collection.get(where={'source': source})
        ent_ids = ents['ids']
        embeddings = []
        for id in ent_ids:
            embed = collection.get(ids=[id], include=["embeddings"])
            embeddings.append(embed['embeddings'][0])
        if len(embeddings) > 1:
            agg_embeddings = embeddings[0]
            for embedding in embeddings[1:]:
                agg_embeddings = np.add(agg_embeddings, embedding)
            
            embed_norm = np.linalg.norm(agg_embeddings)
            norm_embedding = agg_embeddings / embed_norm
        else:
            norm_embedding = embeddings[0]
        output_data.append(norm_embedding)
        output_labels.append(label)

    return np.array(output_data), np.array(output_labels)

if __name__ == "__main__":
    # This will cluster and plot the following data:
    #   - Seed data, all the chunks
    #   - Diverse data, all the chunks
    #   - Seed data, chunks for a document are aggregated and normalized
    #   - Diverse data, chunks for a document are aggregated and normalized

    embedding_function_chroma_codet = ChromaCodet5pEmbedding(CHECKPOINT)
    persistent_client_tok = chromadb.PersistentClient() # default settings
    # this gets the collection since it's already present
    ### Seed Data
    collection_seed_tokens = persistent_client_tok.get_collection("seed_code", embedding_function=embedding_function_chroma_codet)
    print("There are", collection_seed_tokens.count(), "in the collection")
    seed_data, seed_labels = chunked_vecdb_2_aggregated_labeled_embeddings(collection_seed_tokens)
    tsne = TSNE(n_components=2, random_state=42)
    seed_tsne_results = tsne.fit_transform(seed_data)

    figure_save_location = "./figs/seed-token-norm-tsne-results.png"

    plt.figure(figsize=(12, 10))
    for label in np.unique(seed_labels):
        plt.scatter(seed_tsne_results[seed_labels == label, 0], seed_tsne_results[seed_labels == label, 1], alpha=0.7)
    plt.title('Seed')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.legend(["compartmental", "abm"])

    plt.savefig(figure_save_location)
    ### ------------------------------------------------------------------------
    ### token_class_few_full Data
    persistent_client_tok_full = chromadb.PersistentClient()
    collection_full_tokens = persistent_client_tok_full.get_collection("token_class_few_full", embedding_function=embedding_function_chroma_codet)
    full_data, full_labels = chunked_vecdb_2_aggregated_labeled_embeddings(collection_full_tokens)
    full_tsne_results = tsne.fit_transform(full_data)

    figure_save_location = "./figs/full-token-norm-tsne-results.png"

    plt.figure(figsize=(12, 10))
    for label in np.unique(full_labels):
        plt.scatter(full_tsne_results[full_labels == label, 0], full_tsne_results[full_labels == label, 1], alpha=0.7)
    plt.title('400 Diverse Points')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.legend(["compartmental", "abm"])

    plt.savefig(figure_save_location)

    ### ------------------------------------------------------------------------
    ### token_class_few_half Data
    persistent_client_tok_half = chromadb.PersistentClient()
    collection_half_tokens = persistent_client_tok_half.get_collection("token_class_few_half", embedding_function=embedding_function_chroma_codet)
    half_data, half_labels = chunked_vecdb_2_aggregated_labeled_embeddings(collection_half_tokens)
    half_tsne_results = tsne.fit_transform(half_data)

    figure_save_location = "./figs/half-token-norm-tsne-results.png"

    plt.figure(figsize=(12, 10))
    for label in np.unique(half_labels):
        plt.scatter(half_tsne_results[half_labels == label, 0], half_tsne_results[half_labels == label, 1], alpha=0.7)
    plt.title('200 Diverse Points')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.legend(["compartmental", "abm"])

    plt.savefig(figure_save_location)


    ### ------------------------------------------------------------------------
    ### token_class_few_quarter Data
    persistent_client_tok_quarter = chromadb.PersistentClient()
    collection_quarter_tokens = persistent_client_tok_quarter.get_collection("token_class_few_quarter", embedding_function=embedding_function_chroma_codet)
    quarter_data, quarter_labels = chunked_vecdb_2_aggregated_labeled_embeddings(collection_quarter_tokens)
    quarter_tsne_results = tsne.fit_transform(quarter_data)

    figure_save_location = "./figs/quarter-token-norm-tsne-results.png"

    plt.figure(figsize=(12, 10))
    for label in np.unique(quarter_labels):
        plt.scatter(quarter_tsne_results[quarter_labels == label, 0], quarter_tsne_results[quarter_labels == label, 1], alpha=0.7)
    plt.title('100 Diverse Points')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.legend(["compartmental", "abm"])

    plt.savefig(figure_save_location)





