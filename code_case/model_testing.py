import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from torch_geometric.loader import DataLoader as GeometricLoader
import os
from classification_models import TokenClassificationModel, TreeClassificationModel
import tqdm
from langchain_community.document_loaders import DirectoryLoader, TextLoader
import torch_geometric
import chromadb
from chromadb import Documents, EmbeddingFunction

from vector_db import ChromaCodet5pEmbedding, ChromaTreeEmbedding

from tree_model_trainer import EMBED_DIM, IN_CHANNELS, HIDDEN_CHANNELS, OUT_CHANNELS, NODE_CLASSES, COMPRESSED_CLASSES, COMPRESSED_GRAPH_FEATURE, GRAPH_FEATURE
from token_classification_trainer import preprocess_tokenized_dataset, TokenDatasetWrapper
from code_preprocessing import preprocess_tree_query

TOK_EMBED_DIM = 256
TOK_IN_CHANNELS = 72
TOK_HIDDEN_LAY1 = 36
TOK_HIDDEN_LAY2 = 18
GRAPH_CLASS = 1

DEVICE = "cuda"
CHECKPOINT = "Salesforce/codet5p-110m-embedding"
TREE_CHECKPOINT = "./models/tree_ae/tree_ae.pth"
URL = "http://localhost:8000/code2fn/fn-given-filepaths"


GENERATE_DATASETS = False
ENCODE_TEST_DATA = False
FIX_DATA = False
TRAIN_TOKEN = True
TRAIN_TREE = True


LR_RATE = 3e-4
BATCH_SIZE = 32
MAX_EPOCHS = 6

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
    We need to throw the filing into gpt-4o to also benchmark it's performance (likely 100%)
"""

def preprocess_tree_dataset(code_directory, url, model_checkpoint):

    tree_data = []
    raw_code_files = DirectoryLoader(code_directory, glob='*.py', loader_cls=TextLoader).load()
    counter = 1
    for file in raw_code_files:
        encoded_graph = preprocess_tree_query(file.page_content, url, model_checkpoint)
        if encoded_graph is not None:
            if len(encoded_graph) == 1:
                file_name = file.metadata['source']
                label_name = file_name.split("-")[-2]
                label = 0
                if label_name == "abm":
                    label = 1
                tree_data.append((encoded_graph[0], label))
        print(f"file {counter} of {len(raw_code_files)} done")
        counter += 1
    
    dataset = TokenDatasetWrapper(tree_data)
    return dataset


if __name__ == "__main__":

    seed_dataset_tok_path = './dataset/encoded_tokens/seed_tokens.pth'
    diversified_data_tok_path = './dataset/encoded_tokens/diverse_tokens.pth'
    seed_dataset_tree_path = './dataset/encoded_trees/seed_trees.pth'
    diversified_data_tree_path = './dataset/encoded_trees/diverse_trees.pth'

    if FIX_DATA:
        trim_data_tree_directory = "./dataset/trim_data_tree"
        directory_diverse_tree = "./dataset/agentic_data_tree_v2"

        diverse_dataset_tree = preprocess_tree_dataset(directory_diverse_tree, URL, CHECKPOINT)
        trim_seed_dataset_tree = preprocess_tree_dataset(trim_data_tree_directory, URL, CHECKPOINT)

        diversified_data_tree = ConcatDataset([trim_seed_dataset_tree, diverse_dataset_tree])

        torch.save(diversified_data_tree, diversified_data_tree_path)

    if GENERATE_DATASETS:
        # this first chunk of code is just to make sure we trimmed the seed dataset in the same way to create the diversified data
        embedding_function_chroma_codet = ChromaCodet5pEmbedding(CHECKPOINT)
        embedding_function_chroma_tree = ChromaTreeEmbedding(CHECKPOINT, TREE_CHECKPOINT, URL)

        persistent_client_tok = chromadb.PersistentClient() # default settings
        persistent_client_tree = chromadb.PersistentClient() # default settings

        collection_seed_tokens = persistent_client_tok.get_collection("trim_code_tok_v2", embedding_function=embedding_function_chroma_codet)
        collection_seed_tree = persistent_client_tree.get_collection("trim_code_tree_v2", embedding_function=embedding_function_chroma_tree)

        trim_seed_token_vecs = collection_seed_tokens.get(include=["metadatas"])
        trim_seed_tree_vecs = collection_seed_tree.get(include=["metadatas"])

        trimmed_tok_files = []
        trimmed_tree_files = []
        for metadata in trim_seed_token_vecs['metadatas']:
            file = metadata['source']
            trimmed_tok_files.append(file)
        for metadata in trim_seed_tree_vecs['metadatas']:
            file = metadata['source']
            trimmed_tree_files.append(file)

        # now we import the data and construct the 4 datasets, seed (tree and token), token-diverse and tree-diverse

        directory_seed = "./dataset/new_test_code"
        directory_diverse_tok = "./dataset/agentic_data_token_v2"
        directory_diverse_tree = "./dataset/agentic_data_tree_v2"
        trim_data_token_directory = "./dataset/trim_data_token"
        trim_data_tree_directory = "./dataset/trim_data_tree"
        
        # easy data imports
        seed_dataset_tok = preprocess_tokenized_dataset(directory_seed)
        diverse_dataset_tok = preprocess_tokenized_dataset(directory_diverse_tok)
        
        seed_dataset_tree = preprocess_tree_dataset(directory_seed, URL, CHECKPOINT)
        diverse_dataset_tree = preprocess_tree_dataset(directory_diverse_tree, URL, CHECKPOINT)

        # harder data imports
        # check if they have been isolated to new directories, if not, move them over
        if len(os.listdir(trim_data_token_directory)) != len(trimmed_tok_files):
            for filename in trimmed_tok_files:
                last_name = filename.split("/")[-1]
                with open(filename, 'r', encoding='utf-8') as code_file:
                    code_data = code_file.read()
                    code_file.close()
                with open(trim_data_token_directory+"/"+last_name, 'w') as code_writer:
                    code_writer.write(code_data)
                    code_writer.close()
                
        if len(os.listdir(trim_data_tree_directory)) != len(trimmed_tree_files):
            for filename in trimmed_tree_files:
                last_name = filename.split("/")[-1]
                with open(filename, 'r', encoding='utf-8') as code_file:
                    code_data = code_file.read()
                    code_file.close()
                with open(trim_data_tree_directory+"/"+last_name, 'w') as code_writer:
                    code_writer.write(code_data)
                    code_writer.close()
        
        # now we import as usual for the last datasets
        trim_seed_dataset_tok = preprocess_tokenized_dataset(trim_data_token_directory)
        trim_seed_dataset_tree = preprocess_tree_dataset(trim_data_tree_directory, URL, CHECKPOINT)

        diversified_data_tok = ConcatDataset([trim_seed_dataset_tok, diverse_dataset_tok])
        diversified_data_tree = ConcatDataset([trim_seed_dataset_tree, diverse_dataset_tree])

        # finally have all the datasets, diversified_data_tok, diversified_data_tree, seed_dataset_tok, seed_dataset_tree

        # now we save the data so we don't have to generate again
        torch.save(seed_dataset_tok, seed_dataset_tok_path)
        torch.save(diversified_data_tok, diversified_data_tok_path)
        torch.save(seed_dataset_tree, seed_dataset_tree_path)
        torch.save(diversified_data_tree, diversified_data_tree_path)

    if ENCODE_TEST_DATA:
        test_data_directory = "./dataset/test_data"
        test_dataset_tok = preprocess_tokenized_dataset(test_data_directory)
        test_dataset_tree = preprocess_tree_dataset(test_data_directory, URL, CHECKPOINT)

        torch.save(test_dataset_tok, "./dataset/encoded_tokens/test_tokens.pth")
        torch.save(test_dataset_tree, "./dataset/encoded_trees/test_trees.pth")

    # we load the datasets and run the dataloaders
    load_seed_dataset_tok = torch.load(seed_dataset_tok_path)
    load_diversified_data_tok = torch.load(diversified_data_tok_path)
    load_seed_dataset_tree = torch.load(seed_dataset_tree_path)
    load_diversified_data_tree = torch.load(diversified_data_tree_path)
    load_test_dataset_tok = torch.load("./dataset/encoded_tokens/test_tokens.pth")
    load_test_dataset_tree = torch.load("./dataset/encoded_trees/test_trees.pth")

    # need to test and fix the tree data some before putting in loader
    counter = len(load_seed_dataset_tree) - 1
    for (entry, label) in reversed(load_seed_dataset_tree):
        if entry.pe.shape[1] != 12:
            load_seed_dataset_tree.delete_entry(counter)
        counter += -1

    cleaned_dataset = []
    for (entry, label) in reversed(load_diversified_data_tree):
        if type(entry) == torch_geometric.data.data.Data:
            if entry.pe.shape[1] == 12:
                cleaned_dataset.append((entry, label))

    load_diversified_data_tree_clean = TokenDatasetWrapper(cleaned_dataset)

    counter = len(load_test_dataset_tree) - 1
    for (entry, label) in reversed(load_test_dataset_tree):
        if entry.pe.shape[1] != 12:
                load_test_dataset_tree.delete_entry(counter)
        counter += -1

    loader_seed_dataset_tok = DataLoader(dataset=load_seed_dataset_tok, batch_size=BATCH_SIZE, shuffle=True)
    loader_diversified_data_tok = DataLoader(dataset=load_diversified_data_tok, batch_size=BATCH_SIZE, shuffle=True)
    loader_seed_dataset_tree = GeometricLoader(dataset=load_seed_dataset_tree, batch_size=BATCH_SIZE, shuffle=True)
    loader_diversified_data_tree = GeometricLoader(dataset=load_diversified_data_tree_clean, batch_size=BATCH_SIZE, shuffle=True)
    loader_test_data_token = DataLoader(dataset=load_test_dataset_tok, batch_size=len(load_seed_dataset_tok))
    loader_test_data_tree = GeometricLoader(dataset=load_test_dataset_tree, batch_size=len(load_seed_dataset_tree))


    # now we initialize the models 
    if TRAIN_TOKEN:
        model_seed = TokenClassificationModel(TOK_EMBED_DIM, TOK_IN_CHANNELS, TOK_HIDDEN_LAY1, TOK_HIDDEN_LAY2, GRAPH_CLASS).to(DEVICE)
        model_diversified = TokenClassificationModel(TOK_EMBED_DIM, TOK_IN_CHANNELS, TOK_HIDDEN_LAY1, TOK_HIDDEN_LAY2, GRAPH_CLASS).to(DEVICE)
        optimizer_seed = optim.Adam(model_seed.parameters(), lr=LR_RATE) # adam optimizer
        optimizer_diversified = optim.Adam(model_diversified.parameters(), lr=LR_RATE) # adam optimizer
        loss_function = nn.BCELoss()

        model_seed.train()
        model_diversified.train()
        
        for epoch in range(0, MAX_EPOCHS):
            overall_loss = 0
            for batch_idx, (data, label) in enumerate(loader_seed_dataset_tok):
                data = data.to(DEVICE)
                label = label.to(DEVICE)
                optimizer_seed.zero_grad()
                output = model_seed(data)
                loss = loss_function(output.view(label.shape[0]), label.float())
                overall_loss += loss.item()
                loss.backward()
                optimizer_seed.step()
            print("\tSeed Epoch", epoch, "\tSeed Average Loss: ", overall_loss/((batch_idx+1)*BATCH_SIZE))

        # training the model on diversified data
        for epoch in range(0, MAX_EPOCHS):
            overall_loss = 0
            for batch_idx, (data, label) in enumerate(loader_diversified_data_tok):
                data = data.to(DEVICE)
                label = label.to(DEVICE)
                optimizer_diversified.zero_grad()
                output = model_diversified(data)
                loss = loss_function(output.view(label.shape[0]), label.float())
                overall_loss += loss.item()
                loss.backward()
                optimizer_diversified.step()
            print("\tDiversified Epoch", epoch, "\tDiversified Average Loss: ", overall_loss/((batch_idx+1)*BATCH_SIZE)) 

    if TRAIN_TREE:
        # initialize the models 
        model_seed = TreeClassificationModel(EMBED_DIM, IN_CHANNELS, HIDDEN_CHANNELS, OUT_CHANNELS, NODE_CLASSES, COMPRESSED_CLASSES, GRAPH_CLASS).to(DEVICE)
        model_diversified = TreeClassificationModel(EMBED_DIM, IN_CHANNELS, HIDDEN_CHANNELS, OUT_CHANNELS, NODE_CLASSES, COMPRESSED_CLASSES, GRAPH_CLASS).to(DEVICE)
        optimizer_seed = optim.Adam(model_seed.parameters(), lr=LR_RATE) # adam optimizer
        optimizer_diversified = optim.Adam(model_diversified.parameters(), lr=LR_RATE) # adam optimizer
        loss_function = nn.BCELoss()

        model_seed.train()
        model_diversified.train()

        for epoch in range(1, MAX_EPOCHS):
            overall_loss = 0
            for batch_idx, (data, label) in enumerate(loader_seed_dataset_tree):
                # need to use data.shape[0] as batch size in view because dataset no longer evenly divisble by 32
                data = data.to(DEVICE)
                label = label.to(DEVICE)
                optimizer_seed.zero_grad()
                output = model_seed(data.x1, data.x2, data.pe, data.edge_index, data.batch)
                loss = loss_function(output.view(label.shape[0]), label.float())
                overall_loss += loss.item()
                loss.backward()
                optimizer_seed.step()
            print("\tSeed Epoch", epoch + 1, "\tSeed Average Loss: ", overall_loss/((batch_idx+1)*BATCH_SIZE))
    
        for epoch in range(1, MAX_EPOCHS):
            overall_loss = 0
            for batch_idx, (data, label) in enumerate(loader_diversified_data_tree):
                # need to use data.shape[0] as batch size in view because dataset no longer evenly divisble by 32
                data = data.to(DEVICE)
                label = label.to(DEVICE)
                optimizer_diversified.zero_grad()
                output = model_diversified(data.x1, data.x2, data.pe, data.edge_index, data.batch)
                loss = loss_function(output.view(label.shape[0]), label.float())
                overall_loss += loss.item()
                loss.backward()
                optimizer_diversified.step()
            print("\tDiverse Epoch", epoch + 1, "\tDiverse Average Loss: ", overall_loss/((batch_idx+1)*BATCH_SIZE))