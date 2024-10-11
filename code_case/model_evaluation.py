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
from metrics import clean_tree_dataset

from torch import nn, optim
from torch.utils.data import DataLoader
import torch.nn.functional as F


from sentence_transformers import SentenceTransformer

import os
from transformers import AutoModel, AutoTokenizer
from typing import List
import random as rd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from scipy.stats import skew, kurtosis


from tree_encoder_model import GNNModel
import torch
import pickle

from vector_db import ChromaCodet5pEmbedding, ChromaTreeEmbedding, ChromaNomicEmbedding

if __name__ == "__main__":
    BATCH_SIZE = 4
    DEVICE = "cuda:1"
    # load the test datasets
    token_test_data = torch.load("./dataset/test_data/token_dataset.pth")
    nomic_test_data = torch.load("./dataset/test_data/nomic_dataset.pth")
    tree_test_data = torch.load("./dataset/test_data/tree_dataset.pth")

    loader_token_test_data = DataLoader(dataset=token_test_data, batch_size=BATCH_SIZE, shuffle=True)
    loader_nomic_test_data = DataLoader(dataset=nomic_test_data, batch_size=BATCH_SIZE, shuffle=True)
    loader_tree_test_data = GeometricLoader(dataset=tree_test_data, batch_size=BATCH_SIZE, shuffle=True)

    token_dir = "./models/token_models"
    nomic_dir = "./models/nomic_token_models"
    tree_dir = "./models/tree_models"

    # first is to evalue the token based models
    TOK_EMBED_DIM = 256
    TOK_IN_CHANNELS = 72
    TOK_HIDDEN_LAY1 = 36
    TOK_HIDDEN_LAY2 = 18
    GRAPH_CLASS = 1
    model = TokenClassificationModel2(TOK_EMBED_DIM, TOK_IN_CHANNELS, TOK_HIDDEN_LAY1, TOK_HIDDEN_LAY2, GRAPH_CLASS).to(DEVICE)
    last_subdir = ""
    results = {}
    # Walk through the directory
    for dirpath, _dirnames, filenames in os.walk(token_dir):
        #print(f"Current Directory: {dirpath}")
        # List all files in the current directory
        for filename in filenames:
            subdir = dirpath.split("/")[-1]
            local_file = os.path.join(dirpath,filename)
            if subdir != last_subdir:
                results[subdir] = []
                # evaluate test data on first model
                total_incorrect = 0
                model.load_state_dict(torch.load(local_file))
                # Set the model to evaluation mode and disable gradient calculations
                model.eval()
                # run the test set through the model
                with torch.no_grad():
                    for data, label in loader_token_test_data:
                        data = data.to(DEVICE)
                        label = label.to(DEVICE)
                        output = model(data)
                        output_classes = (output >= 0.5).long()
                        incorrect_batch = (abs(output_classes.view(-1) - label)).sum().item()
                        total_incorrect += incorrect_batch
                accuracy = (len(token_test_data) - total_incorrect)/len(token_test_data)
                accuracy_percent = accuracy * 100
                #print(f'Accuracy: {accuracy * 100:.2f}%')
                results[subdir].append(accuracy_percent)
            else:
                # evaluate test on the remaining models of same type and append to the dict
                total_incorrect = 0
                model.load_state_dict(torch.load(local_file))
                # Set the model to evaluation mode and disable gradient calculations
                model.eval()
                # run the test set through the model
                with torch.no_grad():
                    for data, label in loader_token_test_data:
                        data = data.to(DEVICE)
                        label = label.to(DEVICE)
                        output = model(data)
                        output_classes = (output >= 0.5).long()
                        incorrect_batch = (abs(output_classes.view(-1) - label)).sum().item()
                        total_incorrect += incorrect_batch
                accuracy = (len(token_test_data) - total_incorrect)/len(token_test_data)
                accuracy_percent = accuracy * 100
                #print(f'Accuracy: {accuracy * 100:.2f}%')
                results[subdir].append(accuracy_percent)
            last_subdir = subdir
    token_data = pd.DataFrame(results)

    # second is to evalate the nomic token based models
    TOK_EMBED_DIM = 128
    TOK_IN_CHANNELS = 72
    TOK_HIDDEN_LAY1 = 36
    TOK_HIDDEN_LAY2 = 18
    GRAPH_CLASS = 1
    model = TokenClassificationModel(TOK_EMBED_DIM, TOK_IN_CHANNELS, TOK_HIDDEN_LAY1, TOK_HIDDEN_LAY2, GRAPH_CLASS).to(DEVICE)
    last_subdir = ""
    results = {}
    # Walk through the directory
    for dirpath, dirnames, filenames in os.walk(nomic_dir):
        #print(f"Current Directory: {dirpath}")
        # List all files in the current directory
        for filename in filenames:
            subdir = dirpath.split("/")[-1]
            local_file = os.path.join(dirpath,filename)
            if subdir != last_subdir:
                results[subdir] = []
                # evaluate test data on first model
                total_incorrect = 0
                # load the model
                model.load_state_dict(torch.load(local_file))
                # Set the model to evaluation mode and disable gradient calculations
                model.eval()
                # run the test set through the model
                with torch.no_grad():
                    for data, label in loader_nomic_test_data:
                        data = data.to(DEVICE)
                        label = label.to(DEVICE)
                        output = model(data)
                        output_classes = (output >= 0.5).long()
                        incorrect_batch = (abs(output_classes.view(-1) - label)).sum().item()
                        total_incorrect += incorrect_batch
                accuracy = (len(token_test_data) - total_incorrect)/len(token_test_data)
                accuracy_percent = accuracy * 100
                #print(f'Accuracy: {accuracy * 100:.2f}%')
                results[subdir].append(accuracy_percent)
            else:
                # evaluate test on the remaining models of same type and append to the dict
                total_incorrect = 0
                # load the model
                model.load_state_dict(torch.load(local_file))
                # Set the model to evaluation mode and disable gradient calculations
                model.eval()
                # run the test set through the model
                with torch.no_grad():
                    for data, label in loader_nomic_test_data:
                        data = data.to(DEVICE)
                        label = label.to(DEVICE)
                        output = model(data)
                        output_classes = (output >= 0.5).long()
                        incorrect_batch = (abs(output_classes.view(-1) - label)).sum().item()
                        total_incorrect += incorrect_batch
                accuracy = (len(token_test_data) - total_incorrect)/len(token_test_data)
                accuracy_percent = accuracy * 100
                #print(f'Accuracy: {accuracy * 100:.2f}%')
                results[subdir].append(accuracy_percent)
            last_subdir = subdir
    nomic_data = pd.DataFrame(results)

    # lastly is to evaluate the tree based models
    model = TreeClassificationModel(EMBED_DIM, IN_CHANNELS, HIDDEN_CHANNELS, OUT_CHANNELS, NODE_CLASSES, COMPRESSED_CLASSES, GRAPH_CLASS).to(DEVICE)
    last_subdir = ""
    results = {}
    # Walk through the directory
    for dirpath, dirnames, filenames in os.walk(tree_dir):
        #print(f"Current Directory: {dirpath}")
        # List all files in the current directory
        for filename in filenames:
            subdir = dirpath.split("/")[-1]
            local_file = os.path.join(dirpath,filename)
            if subdir != last_subdir:
                results[subdir] = []
                # evaluate test data on first model
                total_incorrect = 0
                model.load_state_dict(torch.load(local_file))
                # Set the model to evaluation mode and disable gradient calculations
                model.eval()
                # run the test set through the model
                with torch.no_grad():
                    for batch_idx, (data, label) in enumerate(loader_tree_test_data):
                        # need to use data.shape[0] as batch size in view incase dataset is not evenly divisble by 32
                        data = data.to(DEVICE)
                        label = label.to(DEVICE)
                        output = model(data.x1, data.x2, data.pe, data.edge_index, data.batch)
                        output_classes = (output >= 0.5).long()
                        incorrect_batch = (abs(output_classes.view(-1) - label)).sum().item()
                        total_incorrect += incorrect_batch
                accuracy = (len(token_test_data) - total_incorrect)/len(token_test_data)
                accuracy_percent = accuracy * 100
                #print(f'Accuracy: {accuracy * 100:.2f}%')
                results[subdir].append(accuracy_percent)
            else:
                # evaluate test on the remaining models of same type and append to the dict
                total_incorrect = 0
                model.load_state_dict(torch.load(local_file))
                # Set the model to evaluation mode and disable gradient calculations
                model.eval()
                # run the test set through the model
                with torch.no_grad():
                    for batch_idx, (data, label) in enumerate(loader_tree_test_data):
                        # need to use data.shape[0] as batch size in view incase dataset is not evenly divisble by 32
                        data = data.to(DEVICE)
                        label = label.to(DEVICE)
                        output = model(data.x1, data.x2, data.pe, data.edge_index, data.batch)
                        output_classes = (output >= 0.5).long()
                        incorrect_batch = (abs(output_classes.view(-1) - label)).sum().item()
                        total_incorrect += incorrect_batch
                accuracy = (len(token_test_data) - total_incorrect)/len(token_test_data)
                accuracy_percent = accuracy * 100
                #print(f'Accuracy: {accuracy * 100:.2f}%')
                results[subdir].append(accuracy_percent)
            last_subdir = subdir
    tree_data = pd.DataFrame(results)

    #print(token_data)
    #print(nomic_data)
    #print(tree_data)

    latex_token = token_data.to_latex(index=False)
    latex_nomic = nomic_data.to_latex(index=False)
    latex_tree = tree_data.to_latex(index=False)

    print(latex_token)
    print(latex_nomic)
    print(latex_tree)
