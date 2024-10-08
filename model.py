import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
from torch_geometric.nn import SAGEConv, GCNConv
from torch_geometric.utils import degree
from torch_geometric.loader import NeighborSampler
from dgl.nn.pytorch import SGConv
import dgl

from sklearn.cluster import KMeans
import numpy as np

from utility import cluster_metrics

import warnings

warnings.filterwarnings("ignore")


class AE(nn.Module):
    def __init__(self, in_dim, latent_dim, base_model):
        super(AE, self).__init__()

        self.encoder1 = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU()
        )
        self.encoder2 = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU()
        )
        if base_model == 'ae':
            self.encoder3 = nn.Linear(128, latent_dim)
        elif base_model == 'vae':
            self.encoder3 = nn.Linear(128, latent_dim * 2)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, in_dim)
        )

        self.base_model = base_model

    def forward(self, x):
        x = self.encoder1(x)

        x = self.encoder2(x)

        z = self.encoder3(x)

        if self.base_model == 'ae':
            x = self.decoder(z)
            return x, z

        elif self.base_model == 'vae':
            mu, logvar = z.chunk(2, dim=1)
            z = reparameterization(mu, logvar)

            x = self.decoder(z)
            return x, z, mu, logvar


class GCN(nn.Module):
    def __init__(self, in_channels, out_channels, data_name, ae_model='ae', sigma=0.5):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, 256)
        self.conv2 = GCNConv(256, 128)
        self.conv3 = GCNConv(128, out_channels)

        ae = AE(in_channels, out_channels, ae_model)
        ae.load_state_dict(torch.load(f'pretrained/{data_name}_{ae_model}_state_dict.pt'))

        self.encoder1 = ae.encoder1
        self.encoder2 = ae.encoder2
        self.encoder3 = ae.encoder3

        self.decoder = ae.decoder

        self.sigma = sigma
        self.ae_model = ae_model

    def forward(self, x, edge_index):
        g = self.conv1(x, edge_index)
        g = F.selu(g)
        g = F.dropout(g, training=self.training)

        x = self.encoder1(x)
        g = self.conv2((1 - self.sigma) * g + self.sigma * x, edge_index)
        g = F.selu(g)
        g = F.dropout(g, training=self.training)

        x = self.encoder2(x)
        g = self.conv3((1 - self.sigma) * g + self.sigma * x, edge_index)
        z = self.encoder3(x)

        g = (1 - self.sigma) * g + self.sigma * z

        if self.ae_model == 'vae':
            mu, logvar = z.chunk(2, dim=1)
            z = reparameterization(mu, logvar)

            x = self.decoder(z)

            g = soft_assignment1(g)

            return x, z, mu, logvar, g
        elif self.ae_model == 'ae':
            x = self.decoder(z)

            g = soft_assignment1(g)

            return x, z, g


class SAGE(nn.Module):
    def __init__(self, in_channels, out_channels, data_name, ae_model='ae', sigma=0.5):
        super(SAGE, self).__init__()
        assert ae_model=='ae', "SAGE currently only supports AE as base model."
        ae = AE(in_channels, out_channels, ae_model)
        ae.load_state_dict(torch.load(f'pretrained/{data_name}_{ae_model}_state_dict.pt'))

        self.convs = torch.nn.ModuleList()
        self.convs.append(SGConv(in_channels, 256))
        self.convs.append(SGConv(256, 128))
        self.convs.append(SGConv(128, out_channels))

        self.encoder = torch.nn.ModuleList()
        self.encoder.append(ae.encoder1)
        self.encoder.append(ae.encoder2)
        self.encoder.append(ae.encoder3)

        self.decoder = ae.decoder

        self.sigma = sigma
        self.ae_model = ae_model

        self.activation = nn.PReLU()

    # def forward(self, x, adjs):
    #     for i, (edge_index, _, size) in enumerate(adjs):
    #         x_target = x[:size[1]]
    #         g = self.convs[i]((x, x_target), edge_index)
    #         z = self.encoder[i](x_target)
    #         x = (1 - self.sigma) * g + self.sigma * z
    #         if i != 2:
    #             # x = F.relu(x)
    #             # x = F.dropout(x, p=0.5, training=self.training)
    #             x = self.activation(x)
    #     x_hat = self.decoder(z)
    #     return x_hat, z, soft_assignment1(x)


def reparameterization(mu, logvar):
    epsilon = torch.randn_like(mu)
    return mu + epsilon * torch.exp(logvar / 2)


def ae_loss(x, x_bar, base_model, mu=None, logvar=None):
    mse = nn.MSELoss()
    if base_model == 'ae':
        return mse(x_bar, x)
    elif base_model == 'vae':
        if mu is None or logvar is None:
            raise ValueError('mu and logvar must be provided for VAE')
        recon_loss = mse(x_bar, x)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + kl_loss
    else:
        raise ValueError('Invalid base model')


def auxiliary_information(output, pseudo_label):
    C = output.float()
    S = pseudo_label.float()

    t1 = torch.matmul(torch.t(S), S)
    t1 = torch.matmul(t1, t1)
    t1 = torch.trace(t1)

    t2 = torch.matmul(torch.t(C), C)
    t2 = torch.matmul(t2, t2)
    t2 = torch.trace(t2)

    t3 = torch.matmul(torch.t(C), S)
    t3 = torch.matmul(t3, torch.t(t3))
    t3 = torch.trace(t3)

    cop_loss = 1 / (output.shape[0] ** 2) * (t1 + t2 - 2 * t3)

    return cop_loss


def modularity_loss(output, num_edges, edge_index, degree, pseudo_label):
    edge_weight = torch.ones(edge_index.size(1)).to(output.device)

    # Calculate the modularity matrix
    x = torch.sparse.mm(torch.sparse_coo_tensor(edge_index, edge_weight,
                                                (output.size(0), output.size(0))).double(), output.double())
    x = torch.matmul(output.t().double(), x)
    x = torch.trace(x)

    # Calculate the degree product
    y = torch.matmul(output.t().double(), degree.double())
    y = (y ** 2).sum()
    y = y / (2 * num_edges)

    m_loss = -((x - y) / (2 * num_edges))

    aux_loss = auxiliary_information(output, pseudo_label)

    return m_loss + 0.2 * aux_loss


def soft_assignment1(g):
    g = g / (g.sum())
    g = (F.tanh(g)) ** 2
    g = F.normalize(g)

    return g


def soft_assignment2(g, landmarks):
    g = 1.0 / (1.0 + torch.sum(
        torch.pow(g.unsqueeze(1) - landmarks, 2), 2))
    g = (g.t() / torch.sum(g, 1)).t()

    return g


def target_distribution(q):
    weight = q ** 2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()


def train_gcn(model, optimizer, dataset, device, epochs, ae_model, num_cluster):
    model.train()

    scheduler = lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.1, total_iters=epochs)

    data = dataset[0]
    data = data.to(device)

    num_nodes = data.x.shape[0]
    num_edges = data.edge_index.shape[1]

    deg = degree(data.edge_index[0], num_nodes=num_nodes).to(device)

    accuracies = []
    nmis = []
    aris = []
    f1s = []

    for epoch in range(epochs):
        optimizer.zero_grad()
        if ae_model == 'ae':
            x, z, g = model(data.x, data.edge_index)
            a_loss = ae_loss(data.x, x, ae_model)
        elif ae_model == 'vae':
            x, z, mu, logvar, g = model(data.x, data.edge_index)
            a_loss = ae_loss(data.x, x, ae_model, mu, logvar)
        else:
            raise ValueError('Invalid base model')

        kmeans = KMeans(n_clusters=num_cluster).fit(g.detach().cpu().numpy())
        landmarks = torch.tensor(kmeans.cluster_centers_).to(device)

        labels = data.y.flatten()
        pseudo_label = F.one_hot(labels, num_classes=num_cluster).float()

        ass2 = soft_assignment2(g, landmarks)

        m_loss = modularity_loss(g, num_edges, data.edge_index, deg, pseudo_label)

        kl_loss = F.kl_div(ass2.log(), target_distribution(ass2))

        loss = a_loss + m_loss + kl_loss
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)

        optimizer.step()
        scheduler.step()

        if epoch % 5 == 0:
            pred_y = kmeans.labels_
            acc, f1, nmi, ari = cluster_metrics(labels.cpu().detach().numpy(), pred_y)
            accuracies.append(acc)
            nmis.append(nmi)
            aris.append(ari)
            f1s.append(f1)
            print(f'ACC: {acc:.4f}\t NMI: {nmi:.4f}\t ARI: {ari:.4f}\t F1: {f1:.4f}')

        print(f'Epoch [{epoch}/{epochs}]\tmodularity_loss: {m_loss.item():.4f}\t'
              f'kl_loss: {kl_loss.item():.4f}\t Loss: {loss.item():.4f}')

    metrics = {'acc': accuracies, 'nmi': nmis, 'ari': aris, 'f1': f1s}
    return metrics


def train_sage(model, optimizer, dataset, device, epochs, ae_model, num_cluster, batch_size):
    model.train()
    scheduler = lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.1, total_iters=epochs)
    data = dataset[0].to(device)

    train_loader = NeighborSampler(data.edge_index, node_idx=None, sizes=[25, 10, 5], batch_size=batch_size, shuffle=True,
                                   num_workers=4)

    for epoch in range(epochs):
        total_loss = 0
        g_ = []
        for batch_size, n_id, adjs in train_loader:
            adjs = [adj.to(device) for adj in adjs]

            # get sub graph
            target_nodes = n_id[:batch_size]
            edge_index = adjs[0].edge_index
            src, dst = edge_index
            mask = (src < batch_size) & (dst < batch_size)
            target_edge_index = edge_index[:, mask]
            num_target_edge_index = target_edge_index.shape[1]
            target_degree = torch.bincount(target_edge_index.flatten(), minlength=batch_size)
            target_label = data.y[target_nodes].flatten()
            target_pseudo_label = F.one_hot(target_label, num_classes=num_cluster).float()

            optimizer.zero_grad()

            x, z, g = model(data.x[n_id], adjs)
            g_.append(g.detach().cpu().numpy())
            a_loss = ae_loss(data.x[target_nodes], x, ae_model)

            m_loss = modularity_loss(g, num_target_edge_index, target_edge_index, target_degree, target_pseudo_label)

            if epoch > 0:
                ass2 = soft_assignment2(g, landmarks)
                kl_loss = F.kl_div(ass2.log(), target_distribution(ass2))
                loss = a_loss + m_loss + kl_loss
            else:
                loss = a_loss + m_loss
            loss.backward()
            total_loss += loss.item()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            optimizer.step()
            scheduler.step()

        kmeans = KMeans(n_clusters=num_cluster).fit(np.vstack(g_))
        landmarks = torch.tensor(kmeans.cluster_centers_).to(device)


        loss = total_loss / len(train_loader)
        print(
            f'Epoch [{epoch}/{epochs}]\t Loss: {loss:.4f}')


def test_gcn(model, dataset, device, ae_model):
    model.eval()
    data = dataset[0]
    data.to(device)
    model.to(device)

    if ae_model == 'ae':
        x, z, g = model(data.x, data.edge_index)
    elif ae_model == 'vae':
        x, z, mu, logvar, g = model(data.x, data.edge_index)
    else:
        raise ValueError('Invalid base model')

    return g.detach().cpu().numpy()


def test_sage(model, dataset, device, batch_size):
    model.eval()
    data = dataset[0].to(device)
    model.to(device)

    test_loader = NeighborSampler(data.edge_index, node_idx=None, sizes=[25, 10, 5], batch_size=batch_size, shuffle=False,
                                  num_workers=4)

    g = []
    for batch_size, n_id, adjs in test_loader:
        adjs = [adj.to(device) for adj in adjs]

        x, z, g_ = model(data.x[n_id], adjs)

        g.append(g_.detach().cpu().numpy())

    return np.vstack(g)


def train_ae(model, optimizer, dataset, device, epochs, base_model):
    model.train()
    for epoch in range(epochs):
        data = dataset[0]
        data = data.to(device)
        optimizer.zero_grad()
        if base_model == 'ae':
            x, z = model(data.x)
            loss = ae_loss(data.x, x, base_model)
        elif base_model == 'vae':
            x, z, mu, logvar = model(data.x)
            loss = ae_loss(data.x, x, base_model, mu, logvar)
        else:
            raise ValueError('Invalid base model')
        loss.backward()
        optimizer.step()
        print(f'Epoch [{epochs}/{epoch}]\t Loss: {loss.item():.4f}')


def train_model(model, optimizer, dataset, device, epochs, ae_model, num_cluster, batch_size):
    if model.__class__.__name__ == 'GCN':
        train_gcn(model, optimizer, dataset, device, epochs, ae_model, num_cluster)
    elif model.__class__.__name__ == 'SAGE':
        train_sage(model, optimizer, dataset, device, epochs, ae_model, num_cluster, batch_size)


def test_model(model, dataset, device, ae_model, batch_size):
    if model.__class__.__name__ == 'GCN':
        return test_gcn(model, dataset, device, ae_model)
    elif model.__class__.__name__ == 'SAGE':
        return test_sage(model, dataset, device, batch_size)
