from codet5p_embedder import embedder
import torch
from torch import nn
from torch.utils.data import DataLoader

def include_graph_positional_encodings(graph):
    """
    This function takes in a graph or tree that has already been preprocessed and adds to each feature vector of a node
    it's corresponding positional encoding. For graphs this is based on the Laplacian Eigenvectors of the graph, selecting
    the lowest frequency modes as well. 

    Args:
        graph (graph): the preprocessed graph in question
    Returns:
        dataset (graph): the preprocessed graph with positional encodings added to it's node's feature vectors. 
    """

    return graph

# construct a new model class here that is a gnn

if __name__ == "__main__":
    print("temp")