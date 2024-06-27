from tree_encoder_model import GNNModel
from code_preprocessing import preprocess_tree, ListDatasetWrapper
import torch
from torch import nn, optim
from torch.utils.data import Dataset
from tqdm import tqdm
from torch_geometric.loader import DataLoader



# CONFIG
# Architecture Configs
EMBED_DIM = 256 # fixed by codet5p
IN_CHANNELS = 36 # empirical, might be too small
HIDDEN_CHANNELS = 36 # empirical, might be too small
OUT_CHANNELS = 36 # empirical, might be too small
NODE_CLASSES = 10 # fixed by number of node states
COMPRESSED_CLASSES = 6 # empirical
COMPRESSED_GRAPH_FEATURE = 24 # empirical, might be too small
GRAPH_FEATURE = 36 # empirical, might be too small

# Training Configs
BATCH_SIZE = 8 # empirical, based on most graphs have different shapes, so hard to batch effectively
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # 'cuda:0' if only one gpu
MAX_EPOCHS = 5
LR_RATE = 3e-4

if __name__ == "__main__":

    # load the data in
    code_directory = "./dataset/test_code"
    fn_directory = "./dataset/test_fn"
    tree_dataset = preprocess_tree(fn_directory, code_directory)
    train_loader = DataLoader(dataset=tree_dataset, batch_size=BATCH_SIZE, shuffle=True) # batch the data

    # load model, optimizer, and loss function
    model = GNNModel(EMBED_DIM, IN_CHANNELS, HIDDEN_CHANNELS, OUT_CHANNELS, NODE_CLASSES, COMPRESSED_CLASSES, COMPRESSED_GRAPH_FEATURE, GRAPH_FEATURE).to(DEVICE) # load model to device
    optimizer = optim.Adam(model.parameters(), lr=LR_RATE) # adam optimizer
    loss_function = nn.MSELoss()
    model.train()

    # training the model
    for epoch in range(1, MAX_EPOCHS):
        overall_loss = 0
        for batch_idx, data in tqdm(enumerate(train_loader)):
            # need to use data.shape[0] as batch size in view because dataset no longer evenly divisble by 32
            data = data.to(DEVICE)
            optimizer.zero_grad()
            recon_batch = model(data.x1, data.x2, data.pe, data.edge_index, data.batch)
            loss = loss_function(recon_batch, data.x1)
            overall_loss += loss.item()
            loss.backward()
            optimizer.step()
        print("\tEpoch", epoch + 1, "\tAverage Loss: ", overall_loss/((batch_idx+1)*BATCH_SIZE))