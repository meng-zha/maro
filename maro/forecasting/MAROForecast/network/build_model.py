from network.backbones import  *

def build_model(conf):
    model = eval(conf.model.backbone.name)(conf)
    return model