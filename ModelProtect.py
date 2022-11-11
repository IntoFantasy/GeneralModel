import torch
import random
from typing import List, Dict, Tuple
from collections import OrderedDict


def noise_add(model, device, eps):
    noise_model = {}
    for key, value in model.items():
        noise_model[key] = value + torch.randn(value.size()).to(device)*eps
    return noise_model


def model_protect(net, device, chosen=0.8):
    num_chosen = int(len(net.state_dict()) * chosen)
    model_dist = OrderedDict({})
    layer_chosen = random.sample(net.state_dict().keys(), num_chosen)
    for key in layer_chosen:
        model_dist[key] = net.state_dict()[key].clone()
    modelProtect = noise_add(model_dist, device, 0.01)
    return modelProtect

