import torch
from sklearn.cluster import KMeans

from model import GCN, SAGE, train_model, test_model
from dataset import get_dataset
from utility import cluster_metrics

import numpy as np

import random
import argparse


args = argparse.ArgumentParser(description='GNN Training')
args.add_argument('--dataset', type=str, default='computers', help='Name of the dataset')
args.add_argument('--out_channels', type=int, default=16, help='Number of output channels')
args.add_argument('--ae_model', type=str, default='ae', help='Base model to use')
args.add_argument('--gcn_model', type=str, default='gcn', help='Base model to use')
args.add_argument('--sigma', type=float, default=0.5, help='influcence of the autoencoder')
args.add_argument('--epochs', type=int, default=300, help='Number of epochs')
args.add_argument('--device', type=str, default='cuda', help='Device to train on')
args.add_argument('--seed', type=int, default=45, help='Random seed')
args.add_argument('--num_clusters', type=int, default=3, help='Number of clusters')
args.add_argument('--noise_level', type=int, default=3, help='Noise level')
args.add_argument('--n_runs', type=int, default=3, help='Number of runs')
args.add_argument("--batch_size", type=int, default=1600, help="Batch size for training")


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
    args.sigma = 0.8
    args.gcn_model = 'gcn'

elif args.dataset == 'ogbn-arxiv':
    args.num_clusters = 40
    args.sigma = 0.5
    args.gcn_model = 'sage'

torch.manual_seed(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)

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

total_acc = []
total_f1 = []
total_nmi = []
total_ari = []

for _ in range(args.n_runs):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    train_model(model, optimizer, dataset, device, args.epochs, args.ae_model, args.num_clusters, args.batch_size)

    torch.save(model.to('cpu').state_dict(), f'results/{args.dataset}_state_dict.pt')

    pred_g = test_model(model, dataset, device, args.ae_model, args.batch_size)

    y_pred = KMeans(n_clusters=args.num_clusters).fit_predict(pred_g)

    data = dataset[0].to('cpu')

    acc, f1, nmi, ari = cluster_metrics(data.y.numpy().flatten(), y_pred)

    total_acc.append(acc)
    total_f1.append(f1)
    total_nmi.append(nmi)
    total_ari.append(ari)



mean_acc = np.mean(total_acc)
deviation_acc = np.std(total_acc)

mean_f1 = np.mean(total_f1)
deviation_f1 = np.std(total_f1)

mean_nmi = np.mean(total_nmi)
deviation_nmi = np.std(total_nmi)

mean_ari = np.mean(total_ari)
deviation_ari = np.std(total_ari)

print(f"ACC: {total_acc}")
print(f"Mean ACC: {mean_acc:.4f} +- {deviation_acc:.4f}")
print(f"F1: {total_f1}")
print(f"Mean F1: {mean_f1:.4f} +- {deviation_f1:.4f}")
print(f"NMI: {total_nmi}")
print(f"Mean NMI: {mean_nmi:.4f} +- {deviation_nmi:.4f}")
print(f"ARI: {total_ari}")
print(f"Mean ARI: {mean_ari:.4f} +- {deviation_ari:.4f}")
