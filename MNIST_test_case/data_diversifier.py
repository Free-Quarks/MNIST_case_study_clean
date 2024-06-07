import torch
from torch import nn
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.kl import kl_divergence
from torch.utils.data import ConcatDataset, DataLoader
from tqdm import tqdm
import pandas as pd
from train import DatasetWrapper, INPUT_DIM
import time
from scipy.stats import multivariate_normal
import numpy as np
from statistics import mean, stdev
import os

# config
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MIN_KLD_THRESHOLD = 6


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


def diversify_data_precomputed(original_data, new_data, percent, diversity_matrix, maintain_size=True):
    """
    This function will take in two dataset and diversify the original with data from the second. 
    Using KLD as a metric of their encoding from a previously trained VAE encoder. This assumes the diversity
    matrix has already been computed. 

    Args:
        original_data (dataset): the original dataset to be diversified
        new_data (dataset): the new dataset from which we will diversify the original
        percent (float): the percent of the new data to add to the original dataset
        diversity_matrix (np-array): the diversity matrix of the KLD cross section between the orignal data and the new data
        maintain_size (bool): should the original dataset maintain the same size (True) or do we simply add more data to it (False). 

    Returns:
        diversified_data (dataset): new diversified dataset
    """

    # number of points to find
    num_of_new_data = int(len(new_data)*percent)

    # get the most diverse points by index and min kld
    min_list = []
    for row in diversity_matrix:
        min_list.append(min(row))

    # assign indexes explicitly
    new_points_idx = []
    for i, kld in enumerate(min_list):
        new_points_idx.append((i, kld))

    # sort by min kld
    new_points_idx.sort(key=lambda tup: tup[1], reverse=True)
    
    # only grab the number of new point indexes needed
    new_points_idx = new_points_idx[0:num_of_new_data]

    # actually grab the data
    new_points = []
    for idx, _kld in new_points_idx:
        new_points.append(new_data[idx])

    # convert it to torch Dataset
    new_points = DatasetWrapper(new_points)

    if maintain_size:
        # need to remove data from original for each new point being added 
        # we will remove based on being below a cut off since that is more efficient than finding the abolsute minimal 
        # KLD overlap of all the original data, cut will be a KLD of 6, based on previous analysis
        diversified_data = original_data

        # load encoding model
        encoding_model = torch.load('./MNIST_test_case/saved_models/vae_model.pt')
        # need to shut off grads to parallelize inference and save space / computation
        for p in encoding_model.parameters():
            p.requires_grad = False
        # move model to device and set up for eval
        encoding_model.to(DEVICE)
        encoding_model.eval() 

        kl_loss = nn.KLDivLoss(reduction="batchmean")

        # now to try out the pointwise KLDloss to see how much faster it is. we will convert each gaussian into a 10x10 matrix representing the distribution. This will be done by using np.linspace to constuct a 10 dimensional vector for each variate in the gaussian (10). The span of the linspace will be dependent on the covariance / stdev of that given dimensionality since our covariance matrix is diagonal, the is essentially decomposing each 10-variate gaussian into 10 sub gaussians, quantizing them and then calculating the pointwise KLD to hopefully be more efficient. 
        
        # first preprocess the gaussians into quantized variants
        # this takes about 150 seconds on kraken, check first if precomputed exists
        if os.path.exists('./MNIST_test_case/saved_distributions/gaussian_data_quant.parquet.snappy'):
            print("Found pre-computed quantized gaussians")
            df_gaussian_data = pd.read_parquet('./MNIST_test_case/saved_distributions/gaussian_data_quant.parquet.snappy').to_numpy()
        else:
            print("Quantizing encoded gaussians")
            gaussian_data = []
            for i, (data, label) in tqdm(enumerate(original_data)):
                mu1, logvar1 = encoding_model.encode(data.view(1, INPUT_DIM).to(DEVICE))
                stdev1 = 0.5*torch.exp(logvar1)
                mu1, stdev1 = mu1.cpu(), stdev1.cpu()
                quant_gaussian = []
                row_sum = [] # need for normalizing the matrix required for correct KLD's
                for l in range(len(mu1[0])):
                    # set domain out to 2x stdev from mean
                    min_x = mu1[0][l]-2*stdev1[0][l]
                    max_x = mu1[0][l]+2*stdev1[0][l]
                    x = np.linspace(min_x, max_x, 10)
                    y = multivariate_normal.pdf(x, mean=mu1[0][l], cov=stdev1[0][l]**2)
                    quant_gaussian.append(y)
                    row_sum.append(sum(y))
                total_sum = sum(row_sum)
                # normalize the matrix
                normalized_quant_gaussian = []
                for row in quant_gaussian:
                    normalized_row = []
                    for element in row:
                        normalized_row.append(element/total_sum)
                    normalized_quant_gaussian.append(normalized_row)

                gaussian_data.append((normalized_quant_gaussian, label))
            
            df_gaussian_data = pd.DataFrame(data=gaussian_data)
            df_gaussian_data.to_parquet('./MNIST_test_case/saved_distributions/gaussian_data_quant.parquet.snappy')
            df_gaussian_data = pd.read_parquet('./MNIST_test_case/saved_distributions/gaussian_data_quant.parquet.snappy').to_numpy()
            
        # push data to GPU
        tf_gaussian_data = []
        for (gaussian, label) in df_gaussian_data:
            cleaned_gaussian = []
            for entry in gaussian:
                cleaned_gaussian.append(entry.tolist())
            cleaned_gaussian = np.array(cleaned_gaussian)
            tf_gaussian = torch.tensor(cleaned_gaussian).to(DEVICE)
            tf_label = torch.tensor(label).to(DEVICE)
            tf_gaussian_data.append((tf_gaussian, tf_label))
        
        # now to calculate the KLD for one triangle of the matrix and remove points below a threshold
        # need to reevaluate what a good threshold is the for the data. 
        removal_list = []
        for i, (gaussian, _) in enumerate(tf_gaussian_data):
            s=time.time()
            klds = []
            for j, (gaussian_1, _)  in enumerate(tf_gaussian_data):
                if j > i:
                    klds.append(kl_loss(gaussian.log(), gaussian_1).cuda().item())
            print(f"mean: {mean(klds)}")
            print(f"stdev: {stdev(klds)}")
            print(f"min KLD: {min(klds)}")
            if min(klds) < MIN_KLD_THRESHOLD:
                removal_list.append(i)
            if len(removal_list) == num_of_new_data:
                break
            print(f"number of points found: {len(removal_list)} out of {num_of_new_data}")
            e=time.time()
            print(f"{e-s}")
        
        # remove the similar points
        removal_list.sort(reverse=True)
        for i in removal_list:
            diversified_data.pop(i)

        # add on the new points
        diversified_data = ConcatDataset([diversified_data, new_points])
        return diversified_data
    else:
        # just add the new data to the original data
        diversified_data = ConcatDataset([original_data, new_points])
        return diversified_data


if __name__ == "__main__":
    train_dataset_9less = torch.load('./dataset/MNIST/modified/train_dataset_9less.pt')
    train_dataset_9 = torch.load('./dataset/MNIST/modified/train_dataset_9.pt')
    diversity_matrix = pd.read_parquet('./MNIST_test_case/saved_distributions/klds_matrix.parquet.snappy').to_numpy()
    diverse_data = diversify_data_precomputed(train_dataset_9less, train_dataset_9, 0.02, diversity_matrix)
    print(len(train_dataset_9less))
    print(len(diverse_data))
    print(f"{len(diverse_data)-len(train_dataset_9less)}")
