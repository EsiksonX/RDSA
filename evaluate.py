import torch
from sklearn.cluster import KMeans

from model import GCN, SAGE, train_model, test_model
from dataset import get_dataset
from utility import cluster_metrics

import numpy as np

import random
import argparse


args = argparse.ArgumentParser(description='GNN Training')
args.add_argument('--dataset', type=str, default='ogbn-arxiv', help='Name of the dataset')
args.add_argument('--out_channels', type=int, default=16, help='Number of output channels')
args.add_argument('--ae_model', type=str, default='ae', help='Base model to use')
args.add_argument('--gcn_model', type=str, default='gcn', help='Base model to use')
args.add_argument('--sigma', type=float, default=0.5, help='influcence of the autoencoder')
args.add_argument('--epochs', type=int, default=300, help='Number of epochs')
args.add_argument('--device', type=str, default='cuda', help='Device to train on')
args.add_argument('--multigpu', type=bool, default=False, help='Use multiple GPUs')
args.add_argument('--seed', type=int, default=45, help='Random seed')
args.add_argument('--num_clusters', type=int, default=3, help='Number of clusters')
args.add_argument('--noise_level', type=int, default=0, help='Noise level')
args.add_argument('--n_runs', type=int, default=1, help='Number of runs')
args.add_argument("--batch_size", type=int, default=20000, help="Batch size for training")


args = args.parse_args()

if args.dataset == 'cora':
    args.num_clusters = 7
    args.sigma = 0.5
    args.gcn_model = 'gcn'

elif args.dataset == 'citeseer':
    args.num_clusters = 6
    args.sigma = 0.5
    args.gcn_model = 'gcn'

elif args.dataset == 'pubmed':
    args.num_clusters = 3
    args.sigma = 0.5
    args.gcn_model = 'gcn'

elif args.dataset == 'computers':
    args.num_clusters = 10
    args.sigma = 0.4
    args.gcn_model = 'gcn'

elif args.dataset == 'photo':
    args.num_clusters = 8
    args.sigma = 0.5
    args.gcn_model = 'gcn'

elif args.dataset == 'reddit':
    args.num_clusters = 41
    args.sigma = 0.5
    args.gcn_model = 'sage'

elif args.dataset == 'ogbn-arxiv':
    args.num_clusters = 40
    args.sigma = 0.7
    args.gcn_model = 'sage'

if args.device == 'cuda' and torch.cuda.is_available():
    device = torch.device('cuda')
    print(f"Using {torch.cuda.get_device_name()}")
else:
    device = torch.device('cpu')
    print("Using CPU")

dataset = get_dataset(args.dataset, args.noise_level)

if args.gcn_model == 'gcn':
    model = GCN(dataset.data.x.shape[1], args.out_channels, args.dataset, args.ae_model, args.sigma).to(device)
elif args.gcn_model == 'sage':
    model = SAGE(dataset.data.x.shape[1], args.out_channels, args.dataset, args.ae_model, args.sigma).to(device)
else:
    raise NotImplementedError(f"Model: {args.gcn_model} not implemented.")

model.load_state_dict(torch.load(f'results/{args.dataset}_state_dict.pt'))

pred_g = test_model(model, dataset, device, args.ae_model, args.batch_size)

y_pred = KMeans(n_clusters=args.num_clusters).fit_predict(pred_g)

data = dataset[0].to('cpu')

acc, f1, nmi, ari = cluster_metrics(data.y.numpy().flatten(), y_pred)

print(f"Accuracy: {acc:.4f}, F1: {f1:.4f}, NMI: {nmi:.4f}, ARI: {ari:.4f}")