import numpy as np
import torch
from torch_geometric.data import Data


def generate_tsp_instance(num_cities):
    coordinates = np.random.rand(num_cities, 2)
    return coordinates

def create_graph_data(coordinates):
    num_nodes = coordinates.shape[0]
    # Create edge index for a fully connected graph
    edge_index = torch.combinations(torch.arange(num_nodes), r=2).t()
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)  # Make edges bidirectional

    # Node features
    x = torch.tensor(coordinates, dtype=torch.float)

    data = Data(x=x, edge_index=edge_index)
    return data