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

from generator import ModelChecker, ModelType
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
import math


from tree_encoder_model import GNNModel
import torch
import pickle

from vector_db import ChromaCodet5pEmbedding, ChromaTreeEmbedding, ChromaNomicEmbedding


import torch

def mcc_metric(y_pred, y_true):
    """
    Compute Matthew's Correlation Coefficient (MCC) for binary classification from sigmoid outputs.
    
    Args:
    - y_true: Ground truth binary labels (tensor of shape [batch_size]).
    - y_pred: Predicted probabilities after sigmoid activation (tensor of shape [batch_size]).
    
    Returns:
    - mcc: The Matthew's Correlation Coefficient (float).
    """
    
    # Apply a threshold to get binary predictions (e.g., threshold of 0.5)
    y_pred = (y_pred >= 0.5).float()
    
    # Compute confusion matrix components, we get a lot of 0's here which break the metric, so +1's are to help with that
    tp = ((y_pred == 1) & (y_true == 1)).sum().item() + 1  # True Positives
    tn = ((y_pred == 0) & (y_true == 0)).sum().item() + 1 # True Negatives
    fp = ((y_pred == 1) & (y_true == 0)).sum().item() + 1 # False Positives
    fn = ((y_pred == 0) & (y_true == 1)).sum().item() + 1 # False Negatives
    
    # Compute numerator and denominator for MCC
    numerator = (tp * tn) - (fp * fn)
    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    
    # To avoid division by zero
    if denominator == 0:
        print("------------")
        print(f"tp: {tp}")
        print(f"fp: {fp}")
        print(f"tn: {tn}")
        print(f"fn: {fn}")
        print("------------")
        return 0.0
    
    # Compute MCC
    mcc = numerator / denominator
    return mcc


if __name__ == "__main__":
    # test data directory
    test_data_dir = "./dataset/test_data"

    BATCH_SIZE = 26
    DEVICE = "cuda:1"
    # load the test datasets
    token_test_data = torch.load("./dataset/test_data/token_dataset.pth")
    nomic_test_data = torch.load("./dataset/test_data/nomic_dataset_fixed.pth")
    tree_test_data = torch.load("./dataset/test_data/tree_dataset.pth")

    loader_token_test_data = DataLoader(dataset=token_test_data, batch_size=BATCH_SIZE, shuffle=True)
    loader_nomic_test_data = DataLoader(dataset=nomic_test_data, batch_size=BATCH_SIZE, shuffle=True)
    loader_tree_test_data = GeometricLoader(dataset=tree_test_data, batch_size=BATCH_SIZE, shuffle=True)

    token_dir = "./models/token_models"
    nomic_dir = "./models/nomic_token_models"
    tree_dir = "./models/tree_models"

    ll_function = nn.BCELoss()

    # first is to evalue the token based models
    TOK_EMBED_DIM = 256
    TOK_IN_CHANNELS = 72
    TOK_HIDDEN_LAY1 = 36
    TOK_HIDDEN_LAY2 = 18
    GRAPH_CLASS = 1
    model = TokenClassificationModel2(TOK_EMBED_DIM, TOK_IN_CHANNELS, TOK_HIDDEN_LAY1, TOK_HIDDEN_LAY2, GRAPH_CLASS).to(DEVICE)
    last_subdir = ""
    results = {}
    results_ll = {}
    results_mcc = {}
    # Walk through the directory
    for dirpath, _dirnames, filenames in os.walk(token_dir):
        #print(f"Current Directory: {dirpath}")
        # List all files in the current directory
        for filename in filenames:
            subdir = dirpath.split("/")[-1]
            local_file = os.path.join(dirpath,filename)
            if subdir != last_subdir:
                results[subdir] = []
                results_ll[subdir] = []
                results_mcc[subdir] = []
                # evaluate test data on first model
                total_incorrect = 0
                ll_total = 0
                mcc_total = 0
                mcc_running = []
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
                        ll_total += ll_function(output.view(label.shape[0]), label.float()).item()
                        mcc_running.append(mcc_metric(output.view(label.shape[0]), label.float()))
                accuracy = (len(token_test_data) - total_incorrect)/len(token_test_data)
                accuracy_percent = accuracy * 100
                mcc_total = sum(mcc_running) / len(mcc_running)
                #print(f'Accuracy: {accuracy * 100:.2f}%')
                results[subdir].append(accuracy_percent)
                results_ll[subdir].append(ll_total)
                results_mcc[subdir].append(mcc_total)

            else:
                # evaluate test on the remaining models of same type and append to the dict
                total_incorrect = 0
                ll_total = 0
                mcc_total = 0
                mcc_running = []
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
                        ll_total += ll_function(output.view(label.shape[0]), label.float()).item()
                        mcc_running.append(mcc_metric(output.view(label.shape[0]), label.float()))
                accuracy = (len(token_test_data) - total_incorrect)/len(token_test_data)
                accuracy_percent = accuracy * 100
                mcc_total = sum(mcc_running) / len(mcc_running)
                #print(f'Accuracy: {accuracy * 100:.2f}%')
                results[subdir].append(accuracy_percent)
                results_ll[subdir].append(ll_total)
                results_mcc[subdir].append(mcc_total)
            last_subdir = subdir
    token_data = pd.DataFrame(results)
    token_data_ll = pd.DataFrame(results_ll)
    token_data_mcc = pd.DataFrame(results_mcc)

    # second is to evalate the nomic token based models
    TOK_EMBED_DIM = 128
    TOK_IN_CHANNELS = 72
    TOK_HIDDEN_LAY1 = 36
    TOK_HIDDEN_LAY2 = 18
    GRAPH_CLASS = 1
    model = TokenClassificationModel(TOK_EMBED_DIM, TOK_IN_CHANNELS, TOK_HIDDEN_LAY1, TOK_HIDDEN_LAY2, GRAPH_CLASS).to(DEVICE)
    last_subdir = ""
    results = {}
    results_ll = {}
    results_mcc = {}
    # Walk through the directory
    for dirpath, dirnames, filenames in os.walk(nomic_dir):
        #print(f"Current Directory: {dirpath}")
        # List all files in the current directory
        for filename in filenames:
            subdir = dirpath.split("/")[-1]
            local_file = os.path.join(dirpath,filename)
            if subdir != last_subdir:
                results[subdir] = []
                results_ll[subdir] = []
                results_mcc[subdir] = []
                # evaluate test data on first model
                total_incorrect = 0
                ll_total = 0
                mcc_total = 0
                mcc_running = []
                # load the model
                model.load_state_dict(torch.load(local_file))
                # Set the model to evaluation mode and disable gradient calculations
                model.eval()
                # run the test set through the model
                with torch.no_grad():
                    for data, label in loader_nomic_test_data:
                        data = data.to(DEVICE)
                        label = label.to(DEVICE)
                        output = model(data.view(label.shape[0], TOK_EMBED_DIM))
                        output_classes = (output >= 0.5).long()
                        incorrect_batch = (abs(output_classes.view(-1) - label)).sum().item()
                        total_incorrect += incorrect_batch
                        ll_total += ll_function(output.view(label.shape[0]), label.float()).item()
                        mcc_running.append(mcc_metric(output.view(label.shape[0]), label.float()))
                accuracy = (len(token_test_data) - total_incorrect)/len(token_test_data)
                accuracy_percent = accuracy * 100
                mcc_total = sum(mcc_running) / len(mcc_running)
                #print(f'Accuracy: {accuracy * 100:.2f}%')
                results[subdir].append(accuracy_percent)
                results_ll[subdir].append(ll_total)
                results_mcc[subdir].append(mcc_total)
            else:
                # evaluate test on the remaining models of same type and append to the dict
                total_incorrect = 0
                ll_total = 0
                mcc_total = 0
                mcc_running = []
                # load the model
                model.load_state_dict(torch.load(local_file))
                # Set the model to evaluation mode and disable gradient calculations
                model.eval()
                # run the test set through the model
                with torch.no_grad():
                    for data, label in loader_nomic_test_data:
                        data = data.to(DEVICE)
                        label = label.to(DEVICE)
                        output = model(data.view(label.shape[0], TOK_EMBED_DIM))
                        output_classes = (output >= 0.5).long()
                        incorrect_batch = (abs(output_classes.view(-1) - label)).sum().item()
                        total_incorrect += incorrect_batch
                        ll_total += ll_function(output.view(label.shape[0]), label.float()).item()
                        mcc_running.append(mcc_metric(output.view(label.shape[0]), label.float()))
                accuracy = (len(token_test_data) - total_incorrect)/len(token_test_data)
                accuracy_percent = accuracy * 100
                mcc_total = sum(mcc_running) / len(mcc_running)
                #print(f'Accuracy: {accuracy * 100:.2f}%')
                results[subdir].append(accuracy_percent)
                results_ll[subdir].append(ll_total)
                results_mcc[subdir].append(mcc_total)
            last_subdir = subdir
    nomic_data = pd.DataFrame(results)
    nomic_data_ll = pd.DataFrame(results_ll)
    nomic_data_mcc = pd.DataFrame(results_mcc)

    # lastly is to evaluate the tree based models
    model = TreeClassificationModel(EMBED_DIM, IN_CHANNELS, HIDDEN_CHANNELS, OUT_CHANNELS, NODE_CLASSES, COMPRESSED_CLASSES, GRAPH_CLASS).to(DEVICE)
    last_subdir = ""
    results = {}
    results_ll = {}
    results_mcc = {}
    # Walk through the directory
    for dirpath, dirnames, filenames in os.walk(tree_dir):
        #print(f"Current Directory: {dirpath}")
        # List all files in the current directory
        for filename in filenames:
            subdir = dirpath.split("/")[-1]
            local_file = os.path.join(dirpath,filename)
            if subdir != last_subdir:
                results[subdir] = []
                results_ll[subdir] = []
                results_mcc[subdir] = []
                # evaluate test data on first model
                total_incorrect = 0
                ll_total = 0
                mcc_total = 0
                mcc_running = []
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
                        ll_total += ll_function(output.view(label.shape[0]), label.float()).item()
                        mcc_running.append(mcc_metric(output.view(label.shape[0]), label.float()))
                accuracy = (len(token_test_data) - total_incorrect)/len(token_test_data)
                accuracy_percent = accuracy * 100
                mcc_total = sum(mcc_running) / len(mcc_running)
                #print(f'Accuracy: {accuracy * 100:.2f}%')
                results[subdir].append(accuracy_percent)
                results_ll[subdir].append(ll_total)
                results_mcc[subdir].append(mcc_total)
            else:
                # evaluate test on the remaining models of same type and append to the dict
                total_incorrect = 0
                ll_total = 0
                mcc_total = 0
                mcc_running = []
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
                        ll_total += ll_function(output.view(label.shape[0]), label.float()).item()
                        mcc_running.append(mcc_metric(output.view(label.shape[0]), label.float()))
                accuracy = (len(token_test_data) - total_incorrect)/len(token_test_data)
                accuracy_percent = accuracy * 100
                mcc_total = sum(mcc_running) / len(mcc_running)
                #print(f'Accuracy: {accuracy * 100:.2f}%')
                results[subdir].append(accuracy_percent)
                results_ll[subdir].append(ll_total)
                results_mcc[subdir].append(mcc_total)
            last_subdir = subdir
    tree_data = pd.DataFrame(results)
    tree_data_ll = pd.DataFrame(results_ll)
    tree_data_mcc = pd.DataFrame(results_mcc)

    # baseline gpt-4o
    raw_data = []
    for filename in os.listdir(test_data_dir):
        if filename.split(".")[-1] == "py":
            c = os.path.join(test_data_dir, filename)
            with open(c, 'r', encoding='utf-8') as code_file:
                code_data = code_file.read()
                code_file.close()
            label_name = filename.split("-")[-2]
            label_encoding = 0
            if label_name == "abm":
                label_encoding = 1
            raw_data.append((code_data, label_encoding))

    model = ChatOpenAI(model="gpt-4o", temperature=0.0)
    data_parser = JsonOutputParser(pydantic_object=ModelChecker)
    data_format_instructions = data_parser.get_format_instructions()
    data_prompt_template = ChatPromptTemplate.from_messages([("system", "You are a helpful assistant. {data_format_instructions}"),("human", "Determine whether the give code is an implementation of a compartmental model or an agent-based model. \n Code:\n  {code}")])
    data_chain = data_prompt_template | model | data_parser

    predictions = []
    truth = []
    for code, label in raw_data:
        data_result = data_chain.invoke({"code": code, "data_format_instructions": data_format_instructions})
        output_label = data_result['model_type']
        output_encoding = 0
        if output_label == ModelType.abm:
            output_encoding = 1
        predictions.append(output_encoding)
        truth.append(label)
    
    diff_vec = abs(np.subtract(predictions,truth))
    print(diff_vec)
    diff_count = sum(diff_vec)
    base_accuracy = (1-(diff_count/len(raw_data))) * 100
    


    #print(token_data)
    #print(nomic_data)
    #print(tree_data)

    latex_token = token_data.to_latex(index=False)
    latex_nomic = nomic_data.to_latex(index=False)
    latex_tree = tree_data.to_latex(index=False)

    latex_token_ll = token_data_ll.to_latex(index=False)
    latex_nomic_ll = nomic_data_ll.to_latex(index=False)
    latex_tree_ll = tree_data_ll.to_latex(index=False)

    latex_token_mcc = token_data_mcc.to_latex(index=False)
    latex_nomic_mcc = nomic_data_mcc.to_latex(index=False)
    latex_tree_mcc = tree_data_mcc.to_latex(index=False)

    print('Accuracy:')
    print(latex_token)
    print(latex_nomic)
    print(latex_tree)
    print(f"baseline: {base_accuracy}")
    print('--------------')
    print('Log Loss:')
    print(latex_token_ll)
    print(latex_nomic_ll)
    print(latex_tree_ll)
    print('--------------')
    print('MCC:')
    print(latex_token_mcc)
    print(latex_nomic_mcc)
    print(latex_tree_mcc)
    print('--------------')
