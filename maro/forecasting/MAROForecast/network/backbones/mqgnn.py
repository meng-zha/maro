import torch
from torch import nn

from network.build_block import build_graph, build_temporal

# Structure of neural network
class MqGNN(nn.Module):
    def __init__(self, conf):
        self.conf = conf
        super(MqGNN, self).__init__()

        self.data_conf = conf.data.sequence_feature
        self.model_conf = conf.model
        self.backbone_conf = self.model_conf.backbone

        self.history_len = self.data_conf.history.dates
        self.history_feature_size = len(self.data_conf.history.features)
        self.future_len = self.data_conf.future.dates
        self.future_feature_size = len(self.data_conf.future.features)

        encoder_input_len = self.model_conf.temporal.self_hidden
        # if temporal model decode for each time step
        if getattr(self.model_conf.temporal, 'output_repeat', False):
            encoder_input_len *= self.history_len
        decoder_input_len = 0
        graph_input_size = 0
        future_index = 0
        history_index = 0
       
        self.future_embeddings = nn.ModuleDict()
        self.history_embeddings = nn.ModuleDict()
        for feature_conf in self.data_conf.features:
            column = feature_conf.column
            # future embeddings
            if column in self.data_conf.future.features:
                if feature_conf.attr == 'categorical':
                    self.future_embeddings[f'{future_index}'] = nn.Embedding(feature_conf.NN_embedding.num, feature_conf.NN_embedding.dim)
                    encoder_input_len += feature_conf.NN_embedding.dim * self.future_len
                    decoder_input_len += feature_conf.NN_embedding.dim
                else:
                    encoder_input_len += self.future_len
                    decoder_input_len += 1
                future_index += 1
            # history embeddings
            if column in self.data_conf.history.features:
                if feature_conf.attr == 'categorical':
                    self.history_embeddings[f'{history_index}'] = nn.Embedding(feature_conf.NN_embedding.num, feature_conf.NN_embedding.dim)
                    graph_input_size += feature_conf.NN_embedding.dim * self.history_len
                else:
                    graph_input_size += self.history_len
                history_index += 1

        # build model
        self.gnn_dict = nn.ModuleDict()
        for gnn_conf in self.model_conf.GNN:
            gnn_conf.in_size = graph_input_size
            gnn_conf.out_size = gnn_conf.self_hidden
            self.gnn_dict[gnn_conf.name] = build_graph(gnn_conf)
            decoder_input_len += gnn_conf.self_hidden
        # if temporal model encode for each time step
        if getattr(self.model_conf.temporal, 'input_split', False):
            self.model_conf.temporal.in_size = graph_input_size//self.history_len
        else:
            self.model_conf.temporal.in_size = graph_input_size
        self.model_conf.temporal.out_size = self.model_conf.temporal.self_hidden
        self.temporal_block = build_temporal(self.model_conf.temporal)

        # Global & Local features
        self.global_local_feature_size = self.backbone_conf.global_local_feature_size
        self.MLP_encoder = nn.Sequential(
            nn.Linear(encoder_input_len, self.global_local_feature_size * (self.future_len + 1)),
            nn.ReLU(),
        )

        # Global & Local decoder
        decoder_input_len += self.global_local_feature_size * 2
        linear_size = [decoder_input_len] + self.backbone_conf.mlp_decoder_hidden + [self.backbone_conf.out_size]
        linear_layers = []
        for in_size, out_size in zip(linear_size[:-1], linear_size[1:]):
            linear_layers.append(nn.Linear(in_size, out_size))
            linear_layers.append(nn.ReLU())
        self.MLP_decoder = nn.Sequential(*linear_layers[:-1])

    def forward(self, data):
        x_history_raw, x_future = data['history_features'], data['future_features']
        B, N, _, _ = x_future.shape
        device = x_future.device

        x_history = []
        for i in range(self.history_feature_size):
            if f'{i}' in self.history_embeddings:
                x = self.history_embeddings[f'{i}'](x_history_raw[:, :, :, i].long())
            else:
                x = x_history_raw[:, :, :, i:i+1].float()
            x_history.append(x)
        x_history = torch.cat(x_history, dim = -1)

        # extract temporal feature
        x_encoder_list = [self.temporal_block(x_history)]
        x_decoder_list = []

        # extract graph features
        for gnn_conf in self.model_conf.GNN:
            graph_feature = data[gnn_conf.graph_name]
            node_num = data['meta_info']['node_num']

            # reshape graph feature
            graph_node_list = self.conf.data.graph_feature.nodes
            graph_shape = []
            for node in graph_node_list:
                graph_shape.append(node_num[node][0].item())
            graph_feature = graph_feature.reshape(B, *graph_shape, *graph_shape)

            # aggregate input feature
            # TODO: more ways to aggregate, max, min, custom, etc.
            all_node_shape = []
            aggreagate_list = []
            repeat_shape = [1]*len(self.conf.data.shared_columns.nodes)
            for node_idx, node in enumerate(self.conf.data.shared_columns.nodes):
                all_node_shape.append(node_num[node][0].item())
                if node not in graph_node_list:
                    aggreagate_list.append(node_idx + 1)
                    repeat_shape[node_idx] = node_num[node][0].item()
            graph_input_feature = x_history.reshape(B, *all_node_shape, -1).mean(dim=aggreagate_list)

            # encode graph features
            decode_feature = self.gnn_dict[gnn_conf.name](graph_input_feature, graph_feature)

            # repeat embedding
            for repeat_idx in aggreagate_list:
                decode_feature = decode_feature.unsqueeze(dim=repeat_idx)
            decode_feature = decode_feature.repeat(1, *repeat_shape, 1)

            x_decoder_list.append(decode_feature.reshape(B, N, 1, -1).repeat(1, 1, self.future_len, 1))

        # embedding future_feature
        for i in range(self.future_feature_size):
            if f'{i}' in self.future_embeddings:
                x = self.future_embeddings[f'{i}'](x_future[:, :, :, i].long())
                x_encoder_list.append(x.reshape(B, N, -1))
                x_decoder_list.append(x)
            else:
                x = x_future[:, :, :, i].float()
                x_encoder_list.append(x.reshape(B, N, -1))
                x_decoder_list.append(x.unsqueeze(-1))

        x_encoder_input = torch.cat(x_encoder_list, dim=-1)
        encoded_feature = self.MLP_encoder(x_encoder_input)

        global_feature, local_feature = (
            encoded_feature[:, :, 0:self.global_local_feature_size],
            encoded_feature[:, :, self.global_local_feature_size:]
        )

        x_decoder_list.append(global_feature.reshape(B, N, 1, -1).repeat(1, 1, self.future_len, 1))
        x_decoder_list.append(local_feature.reshape(B, N, self.future_len, -1))

        x_decoder_input = torch.cat(x_decoder_list, dim=-1)
        output = self.MLP_decoder(x_decoder_input)
        return output
