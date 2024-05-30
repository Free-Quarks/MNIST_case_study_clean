import torch
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.kl import kl_divergence
from tqdm import tqdm
import pandas as pd
from train import DatasetWrapper, INPUT_DIM
import time

# config
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def diversify_data(original_data, new_data, percent, maintain_size=True):
    """
    This function will take in two dataset and diversify the original with data from the second. 
    Using KLD as a metric of their encoding from a previously trained VAE encoder. 

    Args:
        original_data (dataset): the original dataset to be diversified
        new_data (dataset): the new dataset from which we will diversify the original
        percent (float): the percent of the new data to add to the original dataset
        maintain_size (bool): should the original dataset maintain the same size (True) or do we simply add more data to it (False). 

    Returns:
        diversified_data (dataset): new diversified dataset
    """
    # load encoding model
    encoding_model = torch.load('./MNIST_test_case/saved_models/vae_model.pt')
    # need to shut off grads to parallelize inference and save space / computation
    for p in encoding_model.parameters():
        p.requires_grad = False
    
    # move model to device and set up for eval
    encoding_model.to(DEVICE)
    encoding_model.eval() 

    # initialize diversified dataset as original dataset
    diversified_data = original_data

    # now to collect the most diverse data points from the new_data
    num_of_new_data = int(len(new_data)*percent)
    original_encodings = []
    new_encodings = []

    # encode original data
    for data, label in original_data:
        mu, logvar = encoding_model.encode(data.view(1, INPUT_DIM).to(DEVICE))
        original_encodings.append(MultivariateNormal(mu.to(DEVICE), torch.diagflat(torch.exp(logvar)).to(DEVICE)))

    # encode new data
    for data, label in new_data:
        mu, logvar = encoding_model.encode(data.view(1, INPUT_DIM).to(DEVICE))
        new_encodings.append(MultivariateNormal(mu.to(DEVICE), torch.diagflat(torch.exp(logvar)).to(DEVICE)))


    # test of KLD batch calculation on GPU
    print("made it here!")
    kld_vec = []
    s=time.time()
    for i in range(len(original_encodings)):
        kld_vec.append(kl_divergence(original_encodings[i], new_encodings[0]).cuda())
    e=time.time()

    print(type(kld_vec))
    print(len(kld_vec))
    time_per_comp = (e-s)/len(original_encodings)
    print(f"Time per kld comp: {time_per_comp} seconds")
    print(f"Total time: {((e-s)*len(new_encodings))/60} minutes")

    if maintain_size:
        # stub
        return diversified_data
    else:
        # stub
        return diversified_data




if __name__ == "__main__":
    train_dataset_9less = torch.load('./dataset/MNIST/modified/train_dataset_9less.pt')
    train_dataset_9 = torch.load('./dataset/MNIST/modified/train_dataset_9.pt')
    diversify_data(train_dataset_9less, train_dataset_9, 0.1)
