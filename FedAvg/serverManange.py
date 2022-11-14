import torch
import numpy as np
from tqdm import tqdm
import math

seed=0

np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


# 测试集上验证准确率
def accuracy(net, testDataLoader, dev):
    sum_accu = 0
    num = 0
    # 载入测试集
    for data, label in testDataLoader:
        data, label = data.to(dev), label.to(dev)
        preds = net(data)
        preds = torch.argmax(preds, dim=1)
        sum_accu += (preds == label).float().mean()
        num += 1
    return sum_accu / num


# 用于将残缺的模型补全
def model_complement(local_parameters, global_parameters, net):
    complemented = local_parameters
    for key in global_parameters:
        if key not in local_parameters:
            complemented[key] = global_parameters[key]
    net.load_state_dict(complemented, strict=True)


# 计算权重
def logMax(clients_in_com, client_mark):
    log_sum = 0
    client_weight = {}
    for client in clients_in_com:
        log_sum += math.log(client_mark[client] + 1, 4)
    for client in clients_in_com:
        client_weight[client] = math.log(client_mark[client] + 1, 4) / log_sum
    return client_weight


# 与用户进行通讯
def communicate(clients_in_comm, global_parameters, args, myClients, net, loss_func, opti, testDataLoader, dev,
                client_mark):
    accuracy_dict = {}
    # 存储每个用户提交的层数
    layer_client = {}
    client_model = {}
    # 每个Client基于当前模型参数和自己的数据训练并更新模型
    # 返回每个Client更新后的参数
    for client in clients_in_comm:
        # 获取当前Client训练得到的参数
        # local_parameters 得到客户端的局部变量
        local_parameters = myClients.clients_set[client].localUpdate(args['epoch'], args['batchsize'], net,
                                                                     loss_func, opti, global_parameters)
        complemented_model = net
        model_complement(local_parameters, global_parameters, complemented_model)
        accuracy_rate = accuracy(complemented_model, testDataLoader, dev)
        accuracy_dict[client] = accuracy_rate
        print(client + ' accuracy: {}'.format(accuracy_rate))

        client_model[client] = local_parameters
        for key, var in local_parameters.items():
            if key not in layer_client:
                layer_client[key] = [client]
            else:
                layer_client[key].append(client)

    # 取平均值，得到本次通信中Server得到的更新后的模型参数
    mark_on_client(accuracy_dict, client_mark)
    for key in global_parameters:
        if key in layer_client:
            average_parameter = 0
            client_weight = logMax(layer_client[key], client_mark)
            for client, weight in client_weight.items():
                average_parameter += weight * client_model[client][key]
            global_parameters[key] = average_parameter


# 信用分机制
def mark_on_client(clients_accuracy_dict, client_mark):
    sum_accuracy = 0
    for key, var in clients_accuracy_dict.items():
        sum_accuracy += var
    average_accuracy = sum_accuracy / len(clients_accuracy_dict)
    waterLineReward = average_accuracy
    waterLineWarning = 0.5 * average_accuracy
    for key, var in clients_accuracy_dict.items():
        if var <= waterLineWarning:
            print(key + " warning")
            client_mark[key] *= 0.5
        elif var >= waterLineReward:
            client_mark[key] *= 2
    print(client_mark)
