import argparse
import json


def parse_args():
    with open('server_setting.json', 'r') as f:
        data = f.read()
        data = json.loads(data)
        # print(type(data))
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="FedAvg")

    parser.add_argument('-g', '--gpu', type=str, default=data["gpu"], help='gpu id to use(e.g. 0,1,2,3)')

    # 客户端的数量
    parser.add_argument('-nc', '--num_of_clients', type=int, default=data["num_of_clients"], help='numer of the clients')

    # 随机挑选的客户端的数量
    parser.add_argument('-cf', '--cfraction', type=float, default=data["c_fraction"],
                        help='C fraction, 0 means 1 client, 1 means total clients')

    # 训练次数(客户端更新次数)
    parser.add_argument('-E', '--epoch', type=int, default=data["epoch"], help='local train epoch')

    # batchsize大小
    parser.add_argument('-B', '--batchsize', type=int, default=data["batch_size"], help='local train batch size')

    # 模型名称
    parser.add_argument('-mn', '--model_name', type=str, default=data["model_name"], help='the model to train')

    # 学习率
    parser.add_argument('-lr', "--learning_rate", type=float, default=data["learning_rate"], help="learning rate, \
                        use value from origin paper as default")

    parser.add_argument('-dataset', "--dataset", type=str, default=data["dataset"], help="需要训练的数据集")

    # 模型验证频率（通信频率）
    parser.add_argument('-vf', "--val_freq", type=int, default=5, help="model validation frequency(of communications)")

    parser.add_argument('-sf', '--save_freq', type=int, default=20,
                        help='global model save frequency(of communication)')

    # n um_comm 表示通信次数，此处设置为1k
    parser.add_argument('-ncomm', '--num_comm', type=int, default=data["num_comm"], help='number of communications')

    parser.add_argument('-sp', '--save_path', type=str, default='./checkpoints', help='the saving path of checkpoints')

    parser.add_argument('-iid', '--IID', type=int, default=0, help='the way to allocate data to clients')
    parser.add_argument('-afb', '--AdoptForBad', type=int, default=data["adopt_for_bad"], help='adopt for bad')
    return parser
