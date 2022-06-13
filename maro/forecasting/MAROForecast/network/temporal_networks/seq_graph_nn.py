import torch
from torch import nn

from network.graph_networks.graph_nn import GraphNN

class SequenceGraphNN(nn.Module):
    def __init__(self, gnn_conf):
        super(SequenceGraphNN, self).__init__()
        self.gnn = GraphNN(gnn_conf)

    def forward(self, x):
        B, N, L, C = x.shape
        device = x.device

        # Init an Identity Matrix
        adj = torch.zeros(B, N, N).to(device)
        idx = torch.arange(N).to(device)
        adj[:, idx, idx] = 1
        x = self.gnn(x.reshape(B, N, -1), adj)

        return x