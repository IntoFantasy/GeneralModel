import os
import argparse
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from Models import Mnist_2NN, Mnist_CNN
from clients import ClientsGroup, client
from model.WideResNet import WideResNet
import parameter

parser = parameter.parse_args()


def test_mkdir(path):
    if not os.path.isdir(path):
        os.mkdir(path)


if __name__ == "__main__":
    args = parser.parse_args()
    args = args.__dict__

    # -----------------------文件保存-----------------------#
    # 创建结果文件夹
    # test_mkdir("./result")
    # path = os.getcwd()
    # 结果存放test_accuracy中
    test_txt = open("test_accuracy.txt", mode="a")
    # global_parameters_txt = open("global_parameters.txt",mode="a",encoding="utf-8")
    # ----------------------------------------------------#
    # 创建最后的结果
    test_mkdir(args['save_path'])

    os.environ['CUDA_VISIBLE_DEVICES'] = args['gpu']
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    net = None
    # 初始化模型
    # mnist_2nn
    if args['model_name'] == 'mnist_2nn':
        net = Mnist_2NN()
    # mnist_cnn
    elif args['model_name'] == 'mnist_cnn':
        net = Mnist_CNN()
    # ResNet网络
    elif args['model_name'] == 'wideResNet':
        net = WideResNet(depth=28, num_classes=10).to(dev)

    ## 如果有多个GPU
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        net = torch.nn.DataParallel(net)

    # 将Tenor 张量 放在 GPU上
    net = net.to(dev)

    '''
        回头直接放在模型内部
    '''
    # 定义损失函数
    loss_func = F.cross_entropy
    # 优化算法的，随机梯度下降法
    # 使用Adam下降法
    opti = optim.Adam(net.parameters(), lr=args['learning_rate'])

    ## 创建Clients群
    '''
        创建Clients群100个
        
        得到Mnist数据
        
        一共有60000个样本
        100个客户端
        IID：
            我们首先将数据集打乱，然后为每个Client分配600个样本。
        Non-IID：
            我们首先根据数据标签将数据集排序(即MNIST中的数字大小)，
            然后将其划分为200组大小为300的数据切片，然后分给每个Client两个切片。
    '''
    myClients = ClientsGroup('mnist', args['IID'], args['num_of_clients'], dev)
    testDataLoader = myClients.test_data_loader

    # ---------------------------------------以上准备工作已经完成------------------------------------------#
    # 每次随机选取10个Clients
    num_in_comm = int(max(args['num_of_clients'] * args['cfraction'], 1))

    # 得到全局的参数
    global_parameters = {}
    # net.state_dict()  # 获取模型参数以共享

    # 得到每一层中全连接层中的名称fc1.weight
    # 以及权重weights(tenor)
    # 得到网络每一层上
    for key, var in net.state_dict().items():
        global_parameters[key] = var.clone()

    # num_comm 表示通信次数
    for i in range(args['num_comm']):
        print("communicate round {}".format(i + 1))

        # 对随机选的将100个客户端进行随机排序
        order = np.random.permutation(args['num_of_clients'])
        # 生成个客户端
        clients_in_comm = ['client{}'.format(i) for i in order[0:num_in_comm]]

        print("客户端" + str(clients_in_comm))

        sum_parameters = {}
        parameters_num = {}
        # 每个Client基于当前模型参数和自己的数据训练并更新模型
        # 返回每个Client更新后的参数
        for client in tqdm(clients_in_comm):
            # 获取当前Client训练得到的参数
            # 这一行代码表示Client端的训练函数，我们详细展开：
            # local_parameters 得到客户端的局部变量
            local_parameters = myClients.clients_set[client].localUpdate(args['epoch'], args['batchsize'], net,
                                                                         loss_func, opti, global_parameters)
            # 对所有的Client返回的参数累加（最后取平均值）
            for key, var in local_parameters.items():
                sum_parameters[key] = sum_parameters.get(key, 0) + var
                parameters_num[key] = parameters_num.get(key, 0) + 1

            # if sum_parameters is None:
            #     sum_parameters = {}
            #     for key, var in local_parameters.items():
            #         sum_parameters[key] = var.clone()
            # else:
            #     for var in sum_parameters:
            #         sum_parameters[var] = sum_parameters[var] + local_parameters[var]
        # 取平均值，得到本次通信中Server得到的更新后的模型参数
        for key in global_parameters:
            if key in sum_parameters:
                global_parameters[key] = (sum_parameters[key] / parameters_num[key])

        # test_txt.write("communicate round " + str(i + 1) + str('accuracy: {}'.format(sum_accu / num)) + "\n")

        '''
            训练结束之后，我们要通过测试集来验证方法的泛化性，
            注意:虽然训练时，Server没有得到过任何一条数据，但是联邦学习最终的目的
            还是要在Server端学习到一个鲁棒的模型，所以在做测试的时候，是在Server端进行的
        '''
        # with torch.no_grad():
        # 通讯的频率
        # if (i + 1) % args['val_freq'] == 0:
        #  加载Server在最后得到的模型参数
        net.load_state_dict(global_parameters, strict=True)
        sum_accu = 0
        num = 0
        # 载入测试集
        for data, label in testDataLoader:
            data, label = data.to(dev), label.to(dev)
            preds = net(data)
            preds = torch.argmax(preds, dim=1)
            sum_accu += (preds == label).float().mean()
            num += 1
        print("\n" + 'accuracy: {}'.format(sum_accu / num))

        test_txt.write("communicate round " + str(i + 1) + "  ")
        test_txt.write('accuracy: ' + str(float(sum_accu / num)) + "\n")
        # test_txt.close()

        if (i + 1) % args['save_freq'] == 0:
            torch.save(net, os.path.join(args['save_path'],
                                         '{}_num_comm{}_E{}_B{}_lr{}_num_clients{}_cf{}'.format(args['model_name'],
                                                                                                i, args['epoch'],
                                                                                                args['batchsize'],
                                                                                                args['learning_rate'],
                                                                                                args['num_of_clients'],
                                                                                                args['cfraction'])))

    test_txt.close()
