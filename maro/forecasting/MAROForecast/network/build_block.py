from network.graph_networks import *
from network.temporal_networks import *
from network.backbones import *

def build_graph(graph_conf):
    graph_model = eval(graph_conf.network)(graph_conf)
    return graph_model

def build_temporal(temporal_conf):
    temporal_model = eval(temporal_conf.network)(temporal_conf)
    return temporal_model