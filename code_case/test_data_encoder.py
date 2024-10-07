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

from classification_models import TokenClassificationModel, TreeClassificationModel
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


if __name__ == "__main__":
    # token stuff
    token_checkpoint = "Salesforce/codet5p-110m-embedding"
    # nomic stuff
    nomic_checkpoint = "nomic-ai/nomic-embed-text-v1.5"
    matryoshka_dim = 128
    # tree stuff
    tree_checkpoint = "./models/tree_ae/tree_ae.pth"
    url = "http://localhost:8000/code2fn/fn-given-filepaths"

    test_data_dir = "./dataset/test_data"
    device = "cuda"  # for GPU usage or "cpu" for CPU usage

    # walk the directory and get the unencoded inputs and encoded labels, [0:compartmental, 1:abm]
    raw_data = []
    for filename in os.listdir(test_data_dir):
        c = os.path.join(test_data_dir, filename)
        with open(c, 'r', encoding='utf-8') as code_file:
            code_data = code_file.read()
            code_file.close()
        label_name = filename.split("-")[-2]
        label_encoding = 0
        if label_name == "abm":
            label_encoding = 1
        raw_data.append((code_data, label_encoding))

    # token encoding
    tokenizer = AutoTokenizer.from_pretrained(token_checkpoint, trust_remote_code=True)
    token_model = AutoModel.from_pretrained(token_checkpoint, trust_remote_code=True).to(device)
    token_data = []
    for code_data, label in raw_data:
        file_encoding = tokenizer.encode(code_data, truncation=True, return_tensors="pt").to(device)
        embedding = token_model(file_encoding).to("cpu").detach()
        token_data.append((embedding, label))

    nomic_dataset = TokenDatasetWrapper(token_data)
    torch.save(test_data_dir+"token_dataset.pth", nomic_dataset)
    
    # nomic token encoding 
    nomic_model = SentenceTransformer(nomic_checkpoint, trust_remote_code=True)
    nomic_data = []
    for code_data, label in raw_data:
        embedding = nomic_model.encode(code_data, convert_to_tensor=True)
        embedding = embedding[:matryoshka_dim]
        embedding = F.normalize(embedding, p=2, dim=0).to("cpu").detach()
        nomic_data.append((embedding, label))

    nomic_dataset = TokenDatasetWrapper(nomic_data)
    torch.save(test_data_dir+"nomic_dataset.pth", nomic_dataset)

    # tree encoding
    tree_data = []




    tree_dataset = TokenDatasetWrapper(tree_data)
    torch.save(test_data_dir+"tree_dataset.pth", tree_dataset)