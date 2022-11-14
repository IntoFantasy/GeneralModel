import torch
import random
from typing import List, Dict, Tuple
from collections import OrderedDict

seed=0

random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

model_dropout = torch.nn.Dropout(0.1)
random.seed(43)


def noise_add(model, device, eps):
    noise_model = {}
    for key, value in model.items():
        noise_model[key] = value + torch.randn(value.size()).to(device) * eps
    return noise_model


'''
将模型的部分层和部分维度舍去，以达到保护隐私的目的
'''


def model_protect(net, device, chosen=0.8):
    num_chosen = int(len(net.state_dict()) * chosen)
    model_dist = OrderedDict({})
    layer_chosen = random.sample(net.state_dict().keys(), num_chosen)
    for key in layer_chosen:
        model_dist[key] = model_dropout(net.state_dict()[key].clone())
    modelProtect = noise_add(model_dist, device, 0.001)
    return modelProtect
