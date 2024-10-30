import time

import torch
import numpy as np
from torch_geometric.data import Data, Dataset
from sklearn.neighbors import kneighbors_graph

from torch_geometric.datasets import Planetoid, Amazon
from torch_geometric.utils import to_dense_adj, dense_to_sparse
from ogb.nodeproppred import PygNodePropPredDataset


def get_dataset(dataset_name, noise_level=0):
    if dataset_name == 'cora':
        dataset = Planetoid('data', 'cora')
    elif dataset_name == 'citeseer':
        dataset = Planetoid('data', 'citeseer')
    elif dataset_name == 'pubmed':
        dataset = Planetoid('data', 'pubmed')
    elif dataset_name == 'computers':
        dataset = Amazon('data', 'computers')
    elif dataset_name == 'photo':
        dataset = Amazon('data', 'photo')
    elif dataset_name == 'ogbn-arxiv':
        dataset = PygNodePropPredDataset(name='ogbn-arxiv')
    elif dataset_name == 'ogbn-products':
        dataset = PygNodePropPredDataset(name='ogbn-products')
    else:
        raise NotImplementedError(f'Dataset: {dataset_name} not implemented.')
    if noise_level > 0:
        load_noise_edge(dataset, dataset_name, noise_level)
    return dataset


class GraphDataset(Dataset):
    def __init__(self, dataset_name, noise_level=0):
        super(GraphDataset, self).__init__(None, None)

        self.name = dataset_name
        load_path = f'data/{dataset_name}/'

        self.feat = np.load(load_path + 'feat.npy', allow_pickle=True)
        self.label = np.load(load_path + 'label.npy', allow_pickle=True)

        if noise_level > 0:
            adj = np.load(load_path + f'adj_noise{noise_level}.npy', allow_pickle=True)
        else:
            adj = np.load(load_path + 'adj.npy', allow_pickle=True)

        self.edge_indices = np.argwhere(adj > 0).T

        self.data = self._process_data()

        print(f"Dataset: {dataset_name} with noise level {noise_level} loaded.")
        print(f"Number of nodes: {self.data.x.shape[0]}")
        print(f"Number of edges: {self.data.edge_index.shape[1] // 2}")

    def _process_data(self):
        edge_index = torch.tensor(self.edge_indices, dtype=torch.long)
        x = torch.tensor(self.feat, dtype=torch.float)
        y = torch.tensor(self.label, dtype=torch.long)

        data = Data(x=x, y=y, edge_index=edge_index)
        return data

    def __len__(self):
        return self.data.x.shape[0]

    def __getitem__(self, idx):
        return self.data


def load_noise_edge(dataset, dataset_name, noise_level):
    try:
        dataset.data.edge_index = torch.load(f'data/{dataset_name}/edge_index_noise{noise_level}.pt')
    except FileNotFoundError:
        edge_index = dataset[0].edge_index
        num_nodes = dataset[0].x.shape[0]
        node_classes = dataset[0].y
        noisy_edge_index = add_noise_edge(edge_index, noise_level, num_nodes, node_classes)
        dataset.data.edge_index = noisy_edge_index
        torch.save(noisy_edge_index, f'data/{dataset_name}/edge_index_noise{noise_level}.pt')


def construct_knn_graph(data, k, cutoff):
    knn_graph = kneighbors_graph(data, n_neighbors=k, mode='distance')
    adj_matrix = knn_graph.toarray()

    total_edges = np.sum(adj_matrix > 0) // 2

    num_edges_to_cut = int(total_edges * cutoff)
    edge_indices = np.argpartition(adj_matrix.flatten(), -num_edges_to_cut)[-num_edges_to_cut:]
    adj_matrix.flat[edge_indices] = 0

    edge_indices = np.column_stack(np.where(adj_matrix > 0))

    edge_indices = np.unique(np.sort(edge_indices, axis=1), axis=0)

    return edge_indices.T


def add_noise_edge(edge_index, noise_level, num_nodes, node_classes):
    assert isinstance(noise_level, int), "Noise level must be an integer"
    assert 1 <= noise_level <= 3, "Noise level must be between 1 and 3"


    adj_matrix = to_dense_adj(edge_index, max_num_nodes=num_nodes)[0]

    # Calculate the number of edges and noise edges
    num_edges = adj_matrix.sum().item() // 2
    noise_ratios = [0.3, 0.6, 0.9]
    num_noise_edges = int(noise_ratios[noise_level - 1] * num_edges)

    # Identify non-edges
    non_edges = (adj_matrix == 0).nonzero(as_tuple=False)
    non_edges = non_edges[non_edges[:, 0] < non_edges[:, 1]]  # Keep only upper triangular part

    # Filter out non-edges where nodes belong to the same class
    class_diff = node_classes[non_edges[:, 0]] != node_classes[non_edges[:, 1]]
    non_edges = non_edges[class_diff]

    if len(non_edges) < num_noise_edges:
        raise ValueError("Not enough non-edges to add the specified amount of noise")

    # Randomly select non-edges to become noise edges
    noise_edges = non_edges[torch.randperm(len(non_edges))[:num_noise_edges]]

    # Update adjacency matrix
    adj_matrix[noise_edges[:, 0], noise_edges[:, 1]] = 1
    adj_matrix[noise_edges[:, 1], noise_edges[:, 0]] = 1

    # Convert back to edge index
    new_edge_index = dense_to_sparse(adj_matrix)[0]

    print("Original edges: ", num_edges)
    print("Noise edges added: ", num_noise_edges)
    print("Total edges after noise: ", adj_matrix.sum().item() // 2)

    return new_edge_index


if __name__ == '__main__':
    dataset_name = 'cora'
    noise_level = 1
    dataset = get_dataset(dataset_name, noise_level=0)
    edge_index = dataset[0].edge_index
    num_nodes = dataset[0].x.shape[0]
    node_classes = dataset[0].y

    noisy_edge_index = add_noise_edge(edge_index, noise_level, num_nodes, node_classes)

    torch.save(noisy_edge_index, f'data/{dataset_name}/edge_index_noise{noise_level}.pt')
