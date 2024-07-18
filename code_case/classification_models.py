import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, GCNConv, aggr
from tree_encoder_model import MPNNLayer

# config


class TreeClassificationModel(nn.Module):
    def __init__(self, embed_dim, in_channels, hidden_channels, out_channels, node_classes, compressed_classes, graph_classes):
        super(TreeClassificationModel, self).__init__()
        # graph encoder layers
        self.dropout1 = nn.Dropout(p=0.1)
        self.dropout2 = nn.Dropout(p=0.1)
        self.embed_contraction = nn.Linear(embed_dim, in_channels)
        self.mpnn_feature = MPNNLayer(in_channels, hidden_channels)
        self.gcn_feature = GCNConv(hidden_channels, out_channels)
        self.mpnn_class = MPNNLayer(node_classes, compressed_classes)
        self.gcn_class = GCNConv(compressed_classes, out_channels)
        self.attention_nn = nn.Sequential(
            nn.Linear(out_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels)
        )
        self.attention_pooling = aggr.AttentionalAggregation(self.attention_nn)
        self.attention_2_class = nn.Linear(out_channels, graph_classes)

    def forward(self, x1, x2, pe, edge_index, batch):
        # encode the embedding feature dimension
        x1 = self.dropout1(x1)
        x1 = self.embed_contraction(x1)
        x1_res = F.relu(x1)
        # 3x concatenated pe, so we can use smaller pe dim but also not too small of embedding dim
        x1 = x1_res + torch.cat((pe, pe, pe), dim=-1) 
        x1 = self.mpnn_feature(x1, edge_index)
        x1 = F.relu(x1)
        x1 = self.dropout2(x1)
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
        z = F.relu(z)
        z = self.attention_2_class(z)
        z = F.softmax(z)
        return z


class TokenClassificationModel(nn.Module):
    def __init__(self, embed_dim, in_channels, hidden_layer1, hidden_layer2, graph_classes):
        super(TokenClassificationModel, self).__init__()
        self.dropout1 = nn.Dropout(p=0.1)
        self.dropout2 = nn.Dropout(p=0.1)
        self.embed_contraction = nn.Linear(embed_dim, in_channels)
        self.hidden_layer1 = nn.Linear(in_channels, hidden_layer1)
        self.hidden_layer2 = nn.Linear(hidden_layer1, hidden_layer2)
        self.hidden_layer3 = nn.Linear(hidden_layer2, hidden_layer2)
        self.hidden_2_class = nn.Linear(hidden_layer2, graph_classes)

    def forward(self, x):
        x = self.dropout1(x)
        x = self.embed_contraction(x)
        x_res = F.relu(x)
        x = self.hidden_layer1(x_res)
        x = F.relu(x)
        x = self.hidden_layer2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.hidden_layer3(x)
        x = F.relu(x)
        x_res_18 = x_res[:, :, :18]
        x = x + x_res_18
        x = self.hidden_2_class(x)
        z = F.sigmoid(x)
        return z
