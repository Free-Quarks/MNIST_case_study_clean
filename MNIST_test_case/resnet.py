import torch
from torch import nn
from torchvision.models import resnet18
from train import DatasetWrapper # need this implicitly, i know it upsets linters
from torch.utils.data import DataLoader, ConcatDataset, random_split
from torch.utils.tensorboard import SummaryWriter
from data_diversifier import diversify_data_precomputed
import pandas as pd

# config
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # 'cuda:0' if only one gpu
MAX_EPOCHS = 5
BATCH_SIZE = 128
LR_RATE = 3e-4
PERCENT_NEW_DATA = 0.33
NUM_MODELS = 4

# define resnet18 model
class R18(nn.Module):    
    def __init__(self):
        super(R18, self).__init__()

        self.model = resnet18(num_classes=10)

        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1, bias=False)
    
    def forward(self, x):
        return self.model(x)


if __name__ == "__main__":
    # import the data
    train_dataset_9less = torch.load('./dataset/MNIST/modified/train_dataset_9less.pt')
    train_dataset_9 = torch.load('./dataset/MNIST/modified/train_dataset_9.pt')
    train_dataset_all = ConcatDataset([train_dataset_9less, train_dataset_9])

    # subselect all data of same size as 9less dataset
    frac = len(train_dataset_9less)/len(train_dataset_all) # frac of data of 9less and all the data
    generator = torch.Generator().manual_seed(13)
    datasubsets = random_split(train_dataset_all, [frac, 1-frac], generator=generator)
    train_dataset_all_norm = datasubsets[0]

    # get the diversified data, import the precomputed KLD cross section to dave time
    diversity_matrix = pd.read_parquet('./MNIST_test_case/saved_distributions/klds_matrix.parquet.snappy').to_numpy()
    diverse_dataset_larger = diversify_data_precomputed(train_dataset_9less, train_dataset_9, PERCENT_NEW_DATA, diversity_matrix, maintain_size=False)
    #diverse_dataset = diversify_data_precomputed(train_dataset_9less, train_dataset_9, PERCENT_NEW_DATA, diversity_matrix)

    # load the data
    train_loader_9less = DataLoader(dataset=train_dataset_9less, batch_size=BATCH_SIZE, shuffle=True)
    train_loader_all = DataLoader(dataset=train_dataset_all, batch_size=BATCH_SIZE, shuffle=True)
    train_loader_all_norm = DataLoader(dataset=train_dataset_all_norm, batch_size=BATCH_SIZE, shuffle=True)
    train_loader_diverse_larger = DataLoader(dataset=diverse_dataset_larger, batch_size=BATCH_SIZE, shuffle=True)
    #train_loader_diverse = DataLoader(dataset=diverse_dataset, batch_size=BATCH_SIZE, shuffle=True)

    all_loaders = [train_loader_9less, train_loader_all, train_loader_all_norm, train_loader_diverse_larger]
    loader_names = ["9less", "all", "all_norm", "diverse_larger"]

    for j, loader in enumerate(all_loaders):
        for m in range(NUM_MODELS):
            # initialize model and parallelization across GPU's
            model = R18()
            #model = torch.nn.DataParallel(model)
            model.to(DEVICE)

            # set up optmiizer and loss function
            optimizer = torch.optim.Adam(model.parameters(), lr=LR_RATE)
            criterion = nn.CrossEntropyLoss()

            # set up tensorboard writer
            writer = SummaryWriter(log_dir = "./runs/"+loader_names[j], filename_suffix="_"+loader_names[j]+f"_{m}")

            for epoch in range(MAX_EPOCHS):
                model.train()
                running_loss = 0.0

                for i, (inputs, labels) in enumerate(loader):
                    inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    writer.add_scalar("Loss/train", loss, epoch)
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()

                    if i % 10 == 9:  # Print every 10 batches
                        print(f'Epoch [{epoch+1}/{MAX_EPOCHS}], Step [{i+1}/{len(loader)}], Loss: {running_loss / 10:.6f}')
                        running_loss = 0.0

            writer.close()
            print('Finished Training '+loader_names[j]+f": {m}")
            torch.save(model.state_dict(), "./MNIST_test_case/saved_models/"+loader_names[j]+"/"+loader_names[j]+f"_{m}.pt")