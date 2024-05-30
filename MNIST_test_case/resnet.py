import torch
from torch import nn
from torchvision.models import resnet18
from train import DatasetWrapper # need this implicitly, i know it upsets linters
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.tensorboard import SummaryWriter
from data_diversifier import diversify_data_precomputed
import pandas as pd

# config
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # 'cuda:0' if only one gpu
MAX_EPOCHS = 5
BATCH_SIZE = 128
LR_RATE = 3e-4
PERCENT_NEW_DATA = 0.2

# define resnet18 model
class R18(nn.Module):    
    def __init__(self):
        super(R18, self).__init__()

        self.model = resnet18(num_classes=10)

        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1, bias=False)
    
    def forward(self, x):
        return self.model(x)

# import the data
train_dataset_9less = torch.load('./dataset/MNIST/modified/train_dataset_9less.pt')
train_dataset_9 = torch.load('./dataset/MNIST/modified/train_dataset_9.pt')
train_dataset_all = ConcatDataset([train_dataset_9less, train_dataset_9])

# get the diversified data, import the precomputed KLD cross section to dave time
diversity_matrix = pd.read_parquet('./MNIST_test_case/saved_distributions/klds_matrix.parquet.snappy').to_numpy()
diverse_dataset_larger = diversify_data_precomputed(train_dataset_9less, train_dataset_9, PERCENT_NEW_DATA, diversity_matrix, maintain_size=False)
diverse_dataset = diversify_data_precomputed(train_dataset_9less, train_dataset_9, PERCENT_NEW_DATA, diversity_matrix)

# load the data
train_loader_9less = DataLoader(dataset=train_dataset_9less, batch_size=BATCH_SIZE, shuffle=True)
train_loader_all = DataLoader(dataset=train_dataset_all, batch_size=BATCH_SIZE, shuffle=True)
train_loader_diverse_larger = DataLoader(dataset=diverse_dataset_larger, batch_size=BATCH_SIZE, shuffle=True)
train_loader_diverse = DataLoader(dataset=diverse_dataset, batch_size=BATCH_SIZE, shuffle=True)

# initialize model and parallelization across GPU's
model = R18()
#model = torch.nn.DataParallel(model)
model.to(DEVICE)

# set up optmiizer and loss function
optimizer = torch.optim.Adam(model.parameters(), lr=LR_RATE)
criterion = nn.CrossEntropyLoss()

# set up tensorboard writer
writer = SummaryWriter()

for epoch in range(MAX_EPOCHS):
    model.train()
    running_loss = 0.0

    for i, (inputs, labels) in enumerate(train_loader_9less):
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        writer.add_scalar("Loss/train", loss, epoch)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if i % 10 == 9:  # Print every 10 batches
            print(f'Epoch [{epoch+1}/{MAX_EPOCHS}], Step [{i+1}/{len(train_loader_9less)}], Loss: {running_loss / 10:.6f}')
            running_loss = 0.0

writer.flush()
print('Finished Training')
