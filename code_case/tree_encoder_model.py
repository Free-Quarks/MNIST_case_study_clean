import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, GCNConv, aggr

# config
EMBED_DIM = 256 # fixed by codet5p
IN_CHANNELS = 36 # empirical, might be too small
HIDDEN_CHANNELS = 36 # empirical, might be too small
OUT_CHANNELS = 36 # empirical, might be too small
NODE_CLASSES = 10 # fixed by number of node states
COMPRESSED_CLASSES = 6 # empirical
COMPRESSED_GRAPH_FEATURE = 24 # empirical, might be too small
GRAPH_FEATURE = 36 # empirical, might be too small
BATCH_SIZE = 8 # empirical, based on most graphs have different shapes, so hard to batch effectively

# this just a standard linear layer over an additive aggregation of neighboring node vectors
class MPNNLayer(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(MPNNLayer, self).__init__(aggr='add')  # "Add" aggregation (Step 5)
        self.lin = torch.nn.Linear(in_channels, out_channels)

    def message(self, x_j):
        # x_j has shape [E, in_channels]
        return x_j

    def update(self, aggr_out):
        # aggr_out has shape [N, out_channels]
        return self.lin(aggr_out)

    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]
        return self.propagate(edge_index, x=x) # this calls message, then aggregate, then update


class GNNModel(torch.nn.Module):
    def __init__(self, embed_dim, in_channels, hidden_channels, out_channels, node_classes, compressed_classes, uncompressed_graph_feature, graph_feature):
        super(GNNModel, self).__init__()
        # graph encoder layers
        self.embed_contraction = nn.Linear(embed_dim, in_channels)
        self.mpnn_feature = MPNNLayer(in_channels, hidden_channels)
        self.gcn_feature = GCNConv(hidden_channels, out_channels)
        self.mpnn_class = MPNNLayer(node_classes, compressed_classes)
        self.gcn_class = GCNConv(compressed_classes, out_channels)
        self.attention_nn = nn.Sequential(
            nn.Linear(out_channels, uncompressed_graph_feature),
            nn.ReLU(),
            nn.Linear(uncompressed_graph_feature, graph_feature)
        )
        self.attention_pooling = aggr.AttentionalAggregation(self.attention_nn)

        # graph decoder layers
        self.gcn_decoder = GCNConv(graph_feature, out_channels)
        self.mpnn_decoder1 = MPNNLayer(out_channels, in_channels)
        self.mpnn_decoder2 = MPNNLayer(in_channels, in_channels)
        self.mpnn_decoder3 = MPNNLayer(in_channels, embed_dim)

    def encoder(self, x1, x2, pe, edge_index, batch):
        # encode the embedding feature dimension
        x1 = self.embed_contraction(x1)
        x1_res = F.relu(x1)
        # 3x concatenated pe, so we can use smaller pe dim but also not too small of embedding dim
        x1 = x1_res + torch.cat((pe, pe, pe), dim=-1) 
        x1 = self.mpnn_feature(x1, edge_index)
        x1 = F.relu(x1)
        x1 = self.gcn_feature(x1, edge_index)
        x1 = F.relu(x1)
        # encode the node class dimension
        x2 = self.mpnn_class(x2, edge_index)
        x2 = F.relu(x2)
        x2 = self.gcn_class(x2, edge_index)
        x2 = F.relu(x2)

        # combine the two feature vectors and a residual connection to contracted output
        z = x1 + x2 + x1_res # 4x cat'ing node class to match dimensions
        # global attention pooling
        z = self.attention_pooling(z, batch) # batch is an actual attribute of the data object in pytorch geometric
        return z

    def decoder(self, z, pe, edge_index, batch):
        # reconstruct the graph, project z into the dimension of number of nodes
        # recall that pytorch geometric batches graphs by similar node and edge dimensions, 
        # so we can assume that those are consistent in our manipulations

        # batching is not done how I thought, all the node feature vectors are concatenated in a batch
        # we need to use the batch label to allocate the distinct graph encodings accross all nodes
        z1 = torch.stack(list(map(lambda x: z[x], batch)))

        # decode the graph
        z1 = self.gcn_decoder(z1, edge_index)
        z1_res = F.relu(z1)
        # 3x concatenated pe, so we can use smaller pe dim but also not too small of embedding dim
        z1 = z1_res + torch.cat((pe, pe, pe), dim=-1) 
        z1 = self.mpnn_decoder1(z1, edge_index)
        z1 = F.relu(z1)
        z1 = self.mpnn_decoder2(z1, edge_index)
        z1 = F.relu(z1) + z1_res # add residual connection
        z1 = self.mpnn_decoder3(z1, edge_index)
        z1 = F.relu(z1) 
        return z1

    # again, note that batch is an actual attribute of the data object, so will be passed as data.batch
    def forward(self, x1, x2, pe, edge_index, batch):
        z = self.encoder(x1, x2, pe, edge_index, batch)
        x_recon = self.decoder(z, pe, edge_index, batch)
        return x_recon
