from transformers import AutoModel, AutoTokenizer
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset, ConcatDataset
import os
from classification_models import TokenClassificationModel
import random as rd
import tqdm

import chromadb
from chromadb import Documents, EmbeddingFunction

from vector_db import ChromaCodet5pEmbedding, ChromaTreeEmbedding

from tree_model_trainer import EMBED_DIM, IN_CHANNELS, HIDDEN_CHANNELS, OUT_CHANNELS, NODE_CLASSES, COMPRESSED_CLASSES, COMPRESSED_GRAPH_FEATURE, GRAPH_FEATURE

DEVICE = "cuda"
CHECKPOINT = "Salesforce/codet5p-110m-embedding"
TREE_CHECKPOINT = "./models/tree_ae/tree_ae.pth"
URL = "http://localhost:8000/code2fn/fn-given-filepaths"

"""
1)
    The seed data and diversified data can be read can be process normally, but for the diversified data we will need to be constructed pulling from the vector db's

    To construct it we can pull in the diversified data normally and then we need to grab only the documents in the respective trimmed seed 
    data database. This should then lead to the construction 3 datasets and we will then test our 2 models on these three datasets. 

    The three datasets being:
    a) Full seed
    b) trimmed token seed + token diverse
    c) trimmed tree seed + tree diverse

2)
    We need to construct a small dev and majority training dataset.
    We also need to pull in the test data and have it be such that it is labeled, this should be do able by their location in the their
    respective folders.
3) 
    We need to do a slightly more complicated training workflow that takes into account our dev set
    We need to create 3 models for each model
4) 
    We need to run an inference loop to test them on the test set. 
5)
    We need to throw the filing into gpt-4o to also benchmark it's performance (likely 100% correct)
"""

embedding_function_chroma_codet = ChromaCodet5pEmbedding(CHECKPOINT)
embedding_function_chroma_tree = ChromaTreeEmbedding(CHECKPOINT, TREE_CHECKPOINT, URL)

persistent_client_tok = chromadb.PersistentClient() # default settings
persistent_client_tree = chromadb.PersistentClient() # default settings

collection_seed_tokens = persistent_client_tok.get_collection("trim_code_tok_v2", embedding_function=embedding_function_chroma_codet)
collection_seed_tree = persistent_client_tree.get_collection("trim_code_tree_v2", embedding_function=embedding_function_chroma_tree)

trim_seed_token_vecs = collection_seed_tokens.get(include=["embeddings"])
trim_seed_tree_vecs = collection_seed_tree.get(include=["embeddings"])

