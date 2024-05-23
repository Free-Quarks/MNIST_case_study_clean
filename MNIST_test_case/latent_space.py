import torch
from train import INPUT_DIM, DatasetWrapper
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.kl import kl_divergence
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
import pandas as pd

# config
PRINT_DATA_DIST = False
GENERATE_LATENT = False
GENERATE_SAMPLE_KLDS = False
GENERATE_MATRIX_KLDS = False
PLOTTING_SAMPLE9 = False
PLOTTING_9_MINS = True
PLOTTING_1_MINS = True
GENERATE_1_KLDS = False

# need to put some functions here for parallelization since only top level functions can be pickled and pool does that

# parallelized calculation of klds, inner loop
def process_inner(model_load, gaussian_1, i, j, data1):
    # Processing logic for inner loop
    if j > i:
        mu, logvar = model_load.encode(data1.view(1, INPUT_DIM))
        gaussian = MultivariateNormal(mu, torch.diagflat(torch.exp(logvar)))
        return kl_divergence(gaussian, gaussian_1).item()
    else: 
        pass

# Lambda functions can be parallelized annoyingly, need to define a function object to bypass it
class Lambda_pool(object):
    def __init__(self, model_load, gaussian_1, i):
        self.model_load = model_load
        self.gaussian_1 = gaussian_1
        self.i = i
    def __call__(self, x):
        process_inner(self.model_load, self.gaussian_1, self.i, x[0], x[1][0])

# outer loop for kld calculation
def process_outer(model_load, i, data, my_slice):
    mu1, logvar1 = model_load.encode(data.view(1, INPUT_DIM))
    gaussian_1 = MultivariateNormal(mu1, torch.diagflat(torch.exp(logvar1)))
    klds_1 = []
    with Pool() as pool:
        klds_1.append(pool.map(Lambda_pool(model_load, gaussian_1, i), enumerate(my_slice)))
    return [kld for kld in klds_1 if kld != None] # skipped entries get appended as None, need to filter them out


if __name__ == "__main__":
    # load data
    train_dataset_9less = torch.load('./dataset/MNIST/modified/train_dataset_9less.pt')
    train_dataset_9 = torch.load('./dataset/MNIST/modified/train_dataset_9.pt')
    # load model
    model_load = torch.load('./MNIST_test_case/saved_models/vae_model.pt')
    # need to shut off grads to parallelize inference 
    for p in model_load.parameters():
        p.requires_grad = False
    model_load.eval() 

    # prep plotting, for two plots case
    fig, (pl1, pl2) = plt.subplots(1, 2, figsize=(12, 5))

    # create dataset of gaussian parameters of latent space
    if PRINT_DATA_DIST:
        zero = 0
        one = 0
        two = 0
        three = 0
        four = 0
        five = 0
        six = 0
        seven = 0
        eight = 0
        for data, label in train_dataset_9less:
            if label == 0:
                zero+=1
            elif label == 1:
                one+=1
            elif label == 2:
                two+=1
            elif label == 3:
                three+=1
            elif label == 4:
                four+=1
            elif label == 5:
                five+=1
            elif label == 6:
                six+=1
            elif label == 7:
                seven+=1
            elif label == 8:
                eight+=1
            else:
                print("unknown label")
        print(zero)
        print(one)
        print(two)
        print(three)
        print(four)
        print(five)
        print(six)
        print(seven)
        print(eight)
    
    # construct collection of gaussians for the latent space distribution
    latent_gaussians = []
    if GENERATE_LATENT:
        encodings = []
        for data, label in train_dataset_9less:
            mu, logvar = model_load.encode(data.view(1, INPUT_DIM))
            encoding = (mu, torch.diagflat(torch.exp(logvar)))
            encodings.append(encoding)
            latent_gaussians.append((label, MultivariateNormal(encoding[0], encoding[1])))
        torch.save(latent_gaussians, './MNIST_test_case/saved_distributions/gaussians.pt')
    else: 
        latent_gaussians = torch.load('./MNIST_test_case/saved_distributions/gaussians.pt')
    
    # sample of a 9 image to compare to
    mu9, logvar9 = model_load.encode(train_dataset_9[9][0].view(1, INPUT_DIM))
    gaussian_9 = MultivariateNormal(mu9, torch.diagflat(torch.exp(logvar9)))
    
    klds_9 = []
    if GENERATE_SAMPLE_KLDS:
        for label, gaussian in latent_gaussians:
            klds_9.append((label, kl_divergence(gaussian, gaussian_9).item()))
        df_klds_9 = pd.DataFrame(data=klds_9)
        df_klds_9.to_parquet('./MNIST_test_case/saved_distributions/klds.parquet.snappy')
        #torch.save(klds_9, './MNIST_test_case/saved_distributions/klds.pt')
    else:
        #klds_9 = torch.load('./MNIST_test_case/saved_distributions/klds.pt')
        klds_9 = pd.read_parquet('./MNIST_test_case/saved_distributions/klds.parquet.snappy').to_numpy()
    

    # need to convert the torch tensors to np arrays for plotting
    plotting_list = []
    for label, kld in klds_9:
        entry = []
        entry.append(label)
        entry.append(kld)
        plotting_list.append(entry)
    
    # convert to numpy array
    plotting_array = np.asarray(plotting_list)

    print(f"Min of KLD for sample 9: {np.min(plotting_array[:,1])}")
    print(f"Avg. KLD for sample 9: {np.average(plotting_array[:,1])}")

    klds_9_by_9less = []
    if GENERATE_MATRIX_KLDS:
        my_slice = [(data, label) for i, (data, label) in enumerate(train_dataset_9) if i < 10] # just slice since it takes a long time
        for data, _ in tqdm(train_dataset_9):
            mu9, logvar9 = model_load.encode(data.view(1, INPUT_DIM))
            gaussian_9 = MultivariateNormal(mu9, torch.diagflat(torch.exp(logvar9)))
            klds_9 = []
            for _, gaussian in latent_gaussians:
                klds_9.append(kl_divergence(gaussian, gaussian_9).item())
            klds_9_by_9less.append(klds_9)
        df_klds_9_by_9less = pd.DataFrame(data=klds_9_by_9less)
        df_klds_9_by_9less.to_parquet('./MNIST_test_case/saved_distributions/klds_matrix.parquet.snappy')
        #torch.save(klds_9_by_9less, './MNIST_test_case/saved_distributions/klds_matrix.pt')
    else:
        #klds_9_by_9less = torch.load('./MNIST_test_case/saved_distributions/klds_matrix.pt')
        klds_9_by_9less = pd.read_parquet('./MNIST_test_case/saved_distributions/klds_matrix.parquet.snappy').to_numpy()
        

    if PLOTTING_SAMPLE9:
        # plotting
        plt.scatter(x = plotting_array[:,0], y = plotting_array[:,1])
        plt.show()
        # now to plot the kl_divergance of the label 9 datapoint agains the current distribution
    
    if PLOTTING_9_MINS:
        # This constructions a matrix of KLDs comparing every 9 value in the training 9 dataset with every
        # value in the 9less training set, each row is a 9 compared to each non 9
        min_list = []
        for row in klds_9_by_9less:
            min_list.append(min(row))

        pl1.hist(min_list, bins=range(0,120,6))
        pl1.set_title('Min KLD of each \"9\" datapoint')
        pl1.set_xlabel('Min KLD')
        pl1.set_ylabel('Frequency')
    
    # This will compare 1's against themselves and see what the minimum if for this case
    klds_1_by_1 = []
    if PLOTTING_1_MINS:
        if GENERATE_1_KLDS:
            my_slice = [(data, label) for data, label in train_dataset_9less if label == 1] # just slice since it takes a long time
            for i, (data, _) in tqdm(enumerate(my_slice)):
                mu1, logvar1 = model_load.encode(data.view(1, INPUT_DIM))
                gaussian_1 = MultivariateNormal(mu1, torch.diagflat(torch.exp(logvar1)))
                klds_1 = []
                for j, (data1, _)  in enumerate(my_slice):
                    if j > i:
                        mu, logvar = model_load.encode(data1.view(1, INPUT_DIM))
                        gaussian = MultivariateNormal(mu, torch.diagflat(torch.exp(logvar)))
                        klds_1.append(kl_divergence(gaussian, gaussian_1).item())
                klds_1_by_1.append(klds_1)

            savable_slice = klds_1_by_1[:-1] # last entry is empty due to conditional
            df_savable_slice = pd.DataFrame(data=savable_slice)
            df_savable_slice.to_parquet('./MNIST_test_case/saved_distributions/klds_matrix_1s.parquet.snappy')
            #torch.save(savable_slice, './MNIST_test_case/saved_distributions/klds_matrix_1s.pt')
        else:
            #klds_1_by_1 = torch.load('./MNIST_test_case/saved_distributions/klds_matrix_1s.pt')
            klds_1_by_1 = pd.read_parquet('./MNIST_test_case/saved_distributions/klds_matrix_1s.parquet.snappy').to_numpy()

        min_list1 = []
        for row in klds_1_by_1:
            min_list1.append(min(row))

        pl2.hist(min_list1, bins=range(0,120,6))
        pl2.set_title('Min KLD of each \"1\" datapoint')
        pl2.set_xlabel('Min KLD')
        pl2.set_ylabel('Frequency')
    
    
    plt.show()


    # parallelizing the inner loop for speed ups
"""  klds_1_by_1 = []
    if PLOTTING_1_MINS:
        # construct comparison of 1's
        my_slice = [(data.detach(), label) for data, label in train_dataset_9less if label == 1] # just slice since it takes a long time
        for i, (data, _) in enumerate(my_slice):
            print(model_load)
            klds_1_by_1.append(process_outer(model_load, i, data, my_slice))

        torch.save(klds_1_by_1, './MNIST_test_case/saved_distributions/klds_matrix_1s.pt')
        min_list1 = []
        for row in klds_1_by_1:
            min_list1.append(min(row))

        plt.hist(min_list1, bins=20)
        plt.show()
 """

