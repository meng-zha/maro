import torch
import torch.nn.functional as F
from torch import nn

from network.graph_networks.gnn_block import *


class GraphNN(nn.Module):
    def __init__(self, gnn_conf):
        super(GraphNN, self).__init__()

        self.num_layers = gnn_conf.num_layers
        self.num_graphs = gnn_conf.num_graphs
        self.enhance = gnn_conf.enhance
        self.dropout = nn.Dropout(gnn_conf.dropout)
        self.block = eval(gnn_conf.block)

        out_size = gnn_conf.out_size
        if self.enhance:
            out_size = out_size // self.num_layers

        head_out = out_size // self.num_graphs

        self.gnn = nn.ModuleList(
            [
                nn.ModuleList([self.block(gnn_conf.in_size, head_out, gnn_conf) for _ in range(self.num_graphs)])
            ] + [
                nn.ModuleList([self.block(out_size, head_out, gnn_conf) for _ in range(self.num_graphs)])
                for _ in range(self.num_layers - 1)
            ]
        )

    def forward(self, x, *adjs):
        # x: B * N * k
        # adjs: B * N * N
        res = []
        for i in range(self.num_layers):
            x = [
                self.dropout(F.relu(self.gnn[i][j](x, adjs[j])))
                for j in range(self.num_graphs)
            ]
            x = torch.cat(x, dim=-1)  # B * N * (k * num_head) -> B * N * out_size
            res.append(x)

        # res : num_layers, B * N * k
        if self.enhance:
            res = torch.cat(res, dim=-1)  # res : B * N * (k * num_layers)
        else:
            res = res[-1]

        return res
