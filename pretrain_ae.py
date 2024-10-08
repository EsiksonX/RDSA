from dataset import get_dataset

from model import AE, train_ae

import argparse

import torch


args = argparse.ArgumentParser(description='VAE Per-training')
args.add_argument('--dataset', type=str, default='ogbn-arxiv', help='Name of the dataset')
args.add_argument('--base_model', type=str, default='ae', help='Base model to use')
args.add_argument('--latent_dim', type=int, default=16, help='Latent dimension')
args.add_argument('--epochs', type=int, default=1000, help='Number of epochs')
args.add_argument('--device', type=str, default='cuda', help='Device to train on')
args.add_argument('--multigpu', type=bool, default=False, help='Use multiple GPUs')
args = args.parse_args()

dataset = args.dataset
base_model = args.base_model
latent_dim = args.latent_dim
epochs = args.epochs

if args.device == 'cuda' and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

data = get_dataset(dataset)

model = AE(data._data.x.shape[1], latent_dim, base_model).to(device)

if args.multigpu and torch.cuda.device_count() > 1 and device == torch.device('cuda'):
    model = torch.nn.DataParallel(model)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

train_ae(model, optimizer, data, device, epochs, base_model)

torch.save(model.to('cpu').state_dict(), f'pretrained/{dataset}_{base_model}_state_dict.pt')
