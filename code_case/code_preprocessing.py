from transformers import AutoModel, AutoTokenizer
import torch
from torch import nn
from torch.utils.data import DataLoader
import os

# run the following if it complains: >$ export TOKENIZERS_PARALLELISM="true"

def preprocess_tokenized(directory):
    """
    This function takes in the directory of the code that will be processed into a dataset of tokenized sequence inputs.
    These will also be broken into the length of the contextual window required for our embedding model (600 tokens). 

    Args:
        directory (string): the string of the directory of the code to be processed
    Returns:
        dataset (torch.dataset): a dataset of processed code
    """
    # initialize encoder for tokenization
    checkpoint = "Salesforce/codet5p-110m-embedding"
    device = "cuda:0"  # for GPU usage or "cpu" for CPU usage
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)

    # first we read in and encode all the files in the directory
    file_encodings = []
    walker = os.walk(directory)
    for root, _dirs, files in walker:
        for file in files:
            file_path = root+"/"+file
            file_content = ""
            with open(file_path, 'r') as file:
                file_content = file.read()
                file.close()

            # not with the truncation=True argument set as is, this will truncate larger code files to the max possible size, 512 tokens. So any 512 token inputs are likely truncated. 
            file_encoding = tokenizer.encode(file_content, truncation=True, return_tensors="pt").to(device)
            print(file_encoding.size())
            file_encodings.append(file_encoding)
            

    dataset = []
    return dataset


def preprocess_tree(directory):
    """
    This function takes in the directory of the code to be processed into a dataset of code trees. Each node will have a subet of tokens and the feature vector for that node will come from the embedding model we are using. 

    Args:
        directory (string): the string of the directory of code to be processed
    Returns:
        dataset (torch.dataset): a dataset of processed code
    """

    dataset = []
    return dataset

if __name__ == "__main__":
    directory = "./dataset/test_code"
    dataset = preprocess_tokenized(directory)
    print(dataset)