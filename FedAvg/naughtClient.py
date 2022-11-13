from clients import client
import torch.nn as nn
from ModelProtect import *


class NaughtClient(client):
    def __init__(self, trainDataSet, dev):
        super(NaughtClient, self).__init__(trainDataSet, dev)

    def localUpdate(self, localEpoch, localBatchSize, Net, lossFun, opti, global_parameters):
        '''
        参数什么的不重要，
        localUpdate中进行捣乱
        '''
        device = next(Net.parameters()).device
        print(Net.state_dict())
        return model_protect(Net, device)

