import torch
import torchvision.datasets as datasets  # Standard datasets
from tqdm import tqdm
from torch import optim
from model import VariationalAutoEncoder, loss_function
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import random

# Hyperparameters / Config
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INPUT_DIM = 784
H1_DIM = 200
H2_DIM = 60
Z_DIM = 10
NUM_EPOCHS = 20
BATCH_SIZE = 32
LR_RATE = 5e-4

# need a wrapper because all our dataset manipulation breaks the inital dataset import class
class DatasetWrapper(Dataset):
    def __init__(self, data_list):
        """Initialize the dataset with a Python list"""
        self.data_list = data_list

    def __len__(self):
        """Return the total number of data points in the dataset"""
        return len(self.data_list)

    def __getitem__(self, idx):
        """Retrieve a tuple by index that includes both the input and the label"""
        data, label = self.data_list[idx]
        return data, label

if __name__ == "__main__":
    # Dataset Loading
    # transform=transforms.ToTensor() normalizes the pixels to 0 to 1
    dataset = datasets.MNIST(root="dataset/", train=True, transform=transforms.ToTensor(), download=True)  # 60000 datapoints


    # This dictionary will store the data categorized by labels
    data_by_label = {}
    for i, (data, label) in enumerate(dataset):
        if label not in data_by_label:
            data_by_label[label] = []
        data_by_label[label].append(dataset[i])

    # now to collect data by label
    dataset_9 = []
    dataset_9less = []

    for label in data_by_label:
        if label == 9:
            dataset_9.extend(data_by_label[label])
        else:
            dataset_9less.extend(data_by_label[label])

    # now to create test and training datasets
    # test will be ~20% 9's, total test set of 7200 datapoints ~12% of the data, 1400 9's in it
    test_set = []
    train_dataset_9 = []
    for data in dataset_9:
        if len(test_set) < 1400:
            test_set.append(data)
        else:
            train_dataset_9.append(data)

    train_dataset_9less = []
    random.shuffle(dataset_9less) # need to shuffle the data since we are sampling by idx
    random.shuffle(dataset_9less) # shuffle twice for good measure
    for data in dataset_9less:
        if len(test_set) < 7200:
            test_set.append(data)
        else:
            train_dataset_9less.append(data)

    # wrap all our datasets now
    dataset_9 = DatasetWrapper(dataset_9) # all data of label 9
    dataset_9less = DatasetWrapper(dataset_9less) # all data of no labels 9
    train_dataset_9 = DatasetWrapper(train_dataset_9) # training data of label 9
    train_dataset_9less = DatasetWrapper(train_dataset_9less) # training data of no labels 9
    test_set = DatasetWrapper(test_set) # test set

    # train VAE on 9less dataset
    train_loader = DataLoader(dataset=train_dataset_9less, batch_size=BATCH_SIZE, shuffle=True) # batch the data
    model = VariationalAutoEncoder(INPUT_DIM, H1_DIM, H2_DIM, Z_DIM).to(DEVICE) # load model to device
    optimizer = optim.Adam(model.parameters(), lr=LR_RATE) # adam optimizer
    model.train()

    # training the model
    for epoch in range(1, NUM_EPOCHS):
        overall_loss = 0
        for batch_idx, (data, _) in tqdm(enumerate(train_loader)):
            # need to use data.shape[0] as batch size in view because dataset no longer evenly divisble by 32
            data = data.view(data.shape[0], INPUT_DIM).to(DEVICE) 
            optimizer.zero_grad()
            recon_batch, mu, log_var = model(data)
            loss = loss_function(recon_batch, data, mu, log_var)
            overall_loss += loss.item()
            loss.backward()
            optimizer.step()
        print("\tEpoch", epoch + 1, "\tAverage Loss: ", overall_loss/(batch_idx*BATCH_SIZE))

    # save model
    torch.save(model, './MNIST_test_case/saved_models/vae_model.pt')

    # save datasets
    torch.save(dataset_9, './dataset/MNIST/modified/dataset_9.pt')
    torch.save(dataset_9less, './dataset/MNIST/modified/dataset_9less.pt')
    torch.save(train_dataset_9, './dataset/MNIST/modified/train_dataset_9.pt')
    torch.save(train_dataset_9less, './dataset/MNIST/modified/train_dataset_9less.pt')
    torch.save(test_set, './dataset/MNIST/modified/test_set.pt')