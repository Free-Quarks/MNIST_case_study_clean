from transformers import AutoModel, AutoTokenizer
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset, ConcatDataset
import os
from classification_models import TokenClassificationModel
import random as rd
import tqdm


# Training Configs
BATCH_SIZE = 32 # empirical, based on most graphs have different shapes, so hard to batch effectively
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # 'cuda:0' if only one gpu
MAX_EPOCHS = 6
LR_RATE = 3e-4

# model params
EMBED_DIM = 256
IN_CHANNELS = 72
HIDDEN_LAY1 = 36
HIDDEN_LAY2 = 18
GRAPH_CLASS = 1

# Dataset class for to wrap our data
class TokenDatasetWrapper(Dataset):
    def __init__(self, data_list):
        """Initialize the dataset with a Python list"""
        self.data_list = data_list

    def __len__(self):
        """Return the total number of data points in the dataset"""
        return len(self.data_list)

    def __getitem__(self, idx):
        """Retrieve an entry by index that includes only the data, no label"""
        data = self.data_list[idx][0]
        label = self.data_list[idx][1]
        return data, label
    
    def delete_entry(self, idx):
        self.data_list = self.data_list[:idx] + self.data_list[idx+1:]


def preprocess_tokenized_dataset(directory):
    """
    This function takes in the directory of the code that will be processed into a dataset of tokenized sequence inputs.
    It then gets the embedding vector for the code and creates a labeled dataset by processing the label in the filename.

    Args:
        directory (string): the string of the directory of the code to be processed
    Returns:
        dataset (TokenDatasetWrapper): a dataset of processed code
    """
    # initialize encoder for tokenization
    checkpoint = "Salesforce/codet5p-110m-embedding"
    device = "cuda"  # for GPU usage or "cpu" for CPU usage
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
    model = AutoModel.from_pretrained(checkpoint, trust_remote_code=True).to(device)

    # first we read in and encode all the files in the directory
    file_encodings = []
    walker = os.walk(directory)
    for root, _dirs, files in walker:
        for file in files:
            file_path = root+"/"+file
            file_content = ""
            label_name = file.split("-")[-2]
            label = 0
            if label_name == "abm":
                label = 1
            with open(file_path, 'r') as f:
                file_content = f.read()
                f.close()

            # not with the truncation=True argument set as is, this will truncate larger code files to the max possible size, 512 tokens. So any 512 token inputs are truncated. Need a solution for this
            #print("-----------------------------")
            #print(file_path)
            file_encoding = tokenizer.encode(file_content, truncation=True, return_tensors="pt").to(device)
            embedding = model(file_encoding).to("cpu").detach()
            #print(file_encoding.size())
            file_encodings.append((embedding, label))
            
    # wrap our list into a dataset
    dataset = TokenDatasetWrapper(file_encodings)
    return dataset

if __name__ == "__main__":
    # load data
    directory_seed = "./dataset/new_test_code"
    directory_diverse = "./dataset/agentic_data_token"

    seed_dataset = preprocess_tokenized_dataset(directory_seed)
    diverse_dataset = preprocess_tokenized_dataset(directory_diverse)

    # now to construct the two data cases, one using seed data and one with the diverse generated data
    data_to_del = []
    while  len(data_to_del) < len(diverse_dataset):
        temp = data_to_del
        idx = rd.randint(0, len(seed_dataset))
        temp.append(idx)
        data_to_del = list(set(temp))
    
    # now to order the data to prep for removal
    data_to_del.sort(reverse=True)

    # del subset of data, replace with new diverse data
    diversified_data = seed_dataset

    for idx in data_to_del:
        diversified_data.delete_entry(idx)

    diversified_data = ConcatDataset([diversified_data, diverse_dataset])

    # now we set up the dataloaders for our two datasets
    diverse_loader = DataLoader(dataset=diversified_data, batch_size=BATCH_SIZE, shuffle=True)
    seed_loader = DataLoader(dataset=seed_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # load model, optimizer, and loss function
    model_seed = TokenClassificationModel(EMBED_DIM, IN_CHANNELS, HIDDEN_LAY1, HIDDEN_LAY2, GRAPH_CLASS).to(DEVICE)
    model_diversified = TokenClassificationModel(EMBED_DIM, IN_CHANNELS, HIDDEN_LAY1, HIDDEN_LAY2, GRAPH_CLASS).to(DEVICE)
    optimizer_seed = optim.Adam(model_seed.parameters(), lr=LR_RATE) # adam optimizer
    optimizer_diversified = optim.Adam(model_diversified.parameters(), lr=LR_RATE) # adam optimizer
    loss_function = nn.BCELoss()

    model_seed.train()
    model_diversified.train()

    # training the model on seed data
    for epoch in range(0, MAX_EPOCHS):
        overall_loss = 0
        for batch_idx, (data, label) in enumerate(seed_loader):
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
        for batch_idx, (data, label) in enumerate(diverse_loader):
            data = data.to(DEVICE)
            label = label.to(DEVICE)
            optimizer_diversified.zero_grad()
            output = model_diversified(data)
            loss = loss_function(output.view(label.shape[0]), label.float())
            overall_loss += loss.item()
            loss.backward()
            optimizer_diversified.step()
        print("\tDiversified Epoch", epoch, "\tDiversified Average Loss: ", overall_loss/((batch_idx+1)*BATCH_SIZE))   

# 11.26 
# 7.93