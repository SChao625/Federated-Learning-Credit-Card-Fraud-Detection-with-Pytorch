# main.py
from torch.utils.data import Subset

from server import Server
from client import Client
from model import Model
from load_data import *
import copy
import _pickle as cPickle


def Initialization(num, model):
    # 初始化服务器
    server = Server(model=copy.deepcopy(model))
    test_data = Test_Data(datas)
    server.set_test_data(test_data)
    # 初始化客户端
    clients = [Client(id=i, model=copy.deepcopy(model)) for i in range(0, num)]
    # 获取数据并分配
    train_data = Train_Data(datas)
    subset_len = len(train_data) // num
    # 创建 Subset 对象的列表
    subsets = [Subset(train_data, list(range(i * subset_len, (i + 1) * subset_len))) for i in range(num)]
    # 将 Subset 分配给每个 Client
    for client, subset in zip(clients, subsets):
        client.set_trian_data(subset)
    return server, clients


datas = Datas()
#节点数量
node_num = 3
model = Model()
#全局迭代次数
global_steps = 10

# if __name__ == '__main__':
#     server, clients = Initialization(node_num, model)
#     for global_step in range(0, global_steps):
#         for client in clients:
#             client.train(epochs=1)
#         server.average(clients)
#
#         if global_step % 1 == 0:
#             loss, accuracy = server.test()
#             print(f'global_step:{global_step}\tloss:{loss.item()}\taccuracy:{accuracy}')
#             with open('result.txt', 'a') as f:
#                 f.writelines(f'{global_step}\t{loss.item()}\t{accuracy}\n')
#
#     with open('E:/deeplearning/model/1996', 'wb') as f:
#         cPickle.dump(server, f)

    # with open('E:/deeplearning/model/1996', 'rb') as f:
    #    rf = cPickle.load(f)

if __name__ == "__main__":
    with open('E:/deeplearning/model/1994', 'rb') as f4:
        rf4 = cPickle.load(f4)
    with open('E:/deeplearning/model/1995', 'rb') as f5:
        rf5 = cPickle.load(f5)
    with open('E:/deeplearning/model/1996', 'rb') as f6:
        rf6 = cPickle.load(f6)
    rf4_p = rf4.get_model()
    print(rf4_p)
    loss, accuracy = rf4.test()
    print(f'global_step:{global_steps}\tloss:{loss.item()}\taccuracy:{accuracy}')
    with open('result.txt', 'a') as f:
        f.writelines(f'{global_steps}\t{loss.item()}\t{accuracy}\n')