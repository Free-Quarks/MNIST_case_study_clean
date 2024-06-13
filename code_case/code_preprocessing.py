import torch
from torch import nn
from torch.utils.data import DataLoader
import os

def preprocess_tokenized(directory):
    """
    This function takes in the directory of the code that will be processed into a dataset of tokenized sequence inputs.
    These will also be broken into the length of the contextual window required for our embedding model (600 tokens). 

    Args:
        directory (string): the string of the directory of the code to be processed
    Returns:
        dataset (torch.dataset): a dataset of processed code
    """

    walker = os.walk(directory)
    for _root, _dirs, files in walker:
        for file in files:
            print("temp")

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