import math
from abc import abstractmethod

import torch
from torch import nn


class AbsGnnBlock(nn.Module):
    def __init__(self, in_features, out_features, conf):
        # Initialize self.in_features, self.out_features, self.use_residual
        super(AbsGnnBlock, self).__init__()

        self.in_features = in_features
        self.out_features = out_features


        self.use_residual = getattr(conf, "use_residual", False)
        if self.use_residual:
            self.out_features //= 2

    @abstractmethod
    def forward(self, x, adjs):
        raise NotImplementedError

class DenseFastGAT(AbsGnnBlock):
    """Simple GAT layer, similar to https://arxiv.org/abs/1710.10903."""

    def __init__(self, in_features, out_features, conf):
        # Initialize self.in_features, self.out_features, self.use_residual
        super(DenseFastGAT, self).__init__(in_features, out_features, conf)


        self.dropout = getattr(conf, "dropout", 0)
        self.alpha = getattr(conf, "alpha", 0)
        self.normalize_edge_weight = getattr(conf, "normalize_edge_weight", False)

        self.W = nn.Linear(self.in_features, self.out_features)
        nn.init.xavier_uniform_(self.W.weight, gain=1.414)

        if self.use_residual:
            self.Wr = nn.Linear(self.in_features, self.out_features)
            nn.init.xavier_uniform_(self.Wr.weight, gain=1.414)

        self.layernorm = nn.LayerNorm(self.out_features)

        self.Wai = nn.Sequential(nn.Linear(self.out_features, 1))
        nn.init.xavier_uniform_(self.Wai[0].weight, gain=1.414)

        self.Waj = nn.Sequential(nn.Linear(self.out_features, 1))
        nn.init.xavier_uniform_(self.Waj[0].weight, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

        self.small_tensor = None

    def forward(self, x, adjs):
        # x: [B, N, feature_size]
        # adjs: [B, N, N]
        B, N, _ = adjs.shape
        if self.small_tensor is None:
            self.small_tensor = -1e9 * torch.ones(B, N, 1).to(adjs.device)

        z = self.W(x)

        ai = self.Wai(z)
        aj = self.Waj(z)
        ai = torch.cat([ai, torch.ones_like(ai)], dim=2)
        aj = torch.cat([torch.ones_like(aj), aj], dim=2)

        e = self.leakyrelu(torch.matmul(ai, aj.transpose(1, 2)))
        zero_vec = -9e15 * torch.ones_like(e)

        if self.normalize_edge_weight:
            adjs = normalize_adj(adjs)
            attention = torch.where(adjs > 0, adjs * e, zero_vec)
        else:
            attention = torch.where(adjs > 0, e, zero_vec)

        attention = torch.cat([attention, self.small_tensor[:B, :, :]], dim=-1)
        attention = nn.functional.softmax(attention, dim=-1)[:, :, :N]
        attention = nn.functional.dropout(attention, self.dropout, training=self.training)

        inputs_prime = torch.matmul(attention, z)

        if self.use_residual:
            inputs_prime = torch.cat([inputs_prime, self.layernorm(self.Wr(x))], dim=-1)
        self.small_tensor = None

        return inputs_prime

class GraphConvolution(AbsGnnBlock):
    """Simple GCN layer, similar to https://arxiv.org/abs/1609.02907."""

    def __init__(self, in_features, out_features, conf):
        # Initialize self.in_features, self.out_features, self.use_residual
        super(GraphConvolution, self).__init__(in_features, out_features, conf)

        self.weight = nn.Parameter(torch.FloatTensor(self.in_features, self.out_features))

        if hasattr(conf, "bias") and conf.bias:
            self.bias = nn.Parameter(torch.FloatTensor(self.out_features))
        else:
            self.register_parameter("bias", None)

        self.Wr = nn.Linear(self.in_features, self.out_features)
        nn.init.xavier_uniform_(self.Wr.weight, gain=1.414)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, adj):
        # x: N * k
        # adj: N * N (initial nn parameter)
        # matmul(adj, x)
        adj = normalize_adj(adj)
        x = torch.matmul(x, self.weight)
        output = torch.matmul(adj, x)
        if self.bias is not None:
            output = output + self.bias

        if self.use_residual:
            output = torch.cat([output, self.Wr(x)], dim=-1)

        return output

def normalize_adj(adj, diag=True):
    sign = False
    if adj.dim() == 2:
        sign = True
        adj = adj.unsqueeze(0)

    adj = adj.clone()
    B, N, _ = adj.size()

    if diag:
        idx = torch.arange(N, dtype=torch.long, device=adj.device)
        adj[:, idx, idx] = 1

    deg_inv_sqrt = adj.sum(dim=-1).clamp(min=1).pow(-0.5)

    adj = deg_inv_sqrt.unsqueeze(-1) * adj * deg_inv_sqrt.unsqueeze(-2)

    if sign:
        return adj[0]
    return adj