# main.py
from torch.utils.data import Subset, DataLoader
from torch.nn import functional as F
from server import Server
from client import Client
from model import Model
from load_data import *
import copy
import torch


def Initialization(num, model):
    # 初始化服务器
    server = Server(model=copy.deepcopy(model))
    test_data = Test_Data(datas)
    server.set_test_data(test_data)
    # 初始化客户端
    clients = [Client(id=i, model=copy.deepcopy(model)) for i in range(0, num)]
    # 获取数据并分配
    train_data = Train_Data(datas)

    # 个性化分类
    x_train = train_data.get_train()
    df_findloc = pd.DataFrame(x_train)
    index_list_0 = df_findloc[(df_findloc[19] == 0)].index.tolist()
    index_list_1 = df_findloc[(df_findloc[19] == 1)].index.tolist()
    subsets = [Subset(train_data, list(index_list_0)), Subset(train_data, list(index_list_1))]

    # 创建 Subset 对象的列表
    # subset_len = len(train_data) // num
    # subsets = [Subset(train_data, list(range(i * subset_len, (i + 1) * subset_len))) for i in range(num)]
    # 将 Subset 分配给每个 Client
    for client, subset in zip(clients, subsets):
        client.set_trian_data(subset)
    return server, clients


datas = Datas()
# 节点数量
node_num = 16
model = Model()
# 全局迭代次数
global_steps = 10


if __name__ == '__main__':
    print("start")
    server, clients = Initialization(node_num, model)
    for global_step in range(0, global_steps):
        for client in clients:
            client.train(epochs=1)
        server.average(clients)

        if global_step % 1 == 0:
            loss, accuracy = server.test()
            print(f'global_step:{global_step}\tloss:{loss.item()}\taccuracy:{accuracy}')
            with open('result.txt', 'a') as f:
                f.writelines(f'{global_step}\t{loss.item()}\t{accuracy}\n')

    torch.save(server.get_model(), 'E:/deeplearning/model/all_personal')

# 聚合参数
def aggregate_parameters(state_dicts, weights):
    aggregated_params = {}
    for key in state_dicts[0].keys():
        aggregated_params[key] = sum(w * d[key] for w, d in zip(weights, state_dicts))
    return aggregated_params


def test(global_model, dataloader):
    global_model.eval()
    total_samples = 0
    correct_predictions = 0
    lossfn = F.binary_cross_entropy_with_logits

    for (x, y) in dataloader:
        # 获取模型预测结果
        logits = global_model(x)
        # 计算损失
        loss = lossfn(logits, y)
        # 对 logits 应用 sigmoid 激活函数
        probabilities = torch.sigmoid(logits)
        # 将概率值转换为预测标签
        predicted_labels = (probabilities > 0.5).float()
        # 计算准确率
        total_samples += y.size(0)
        correct_predictions += (predicted_labels == y).sum().item()
        accuracy = correct_predictions / total_samples
        return loss, accuracy


# if __name__ == "__main__":
#     rf4_m = Model()
#     rf5_m = Model()
#     rf6_m = Model()
#     rfall_m = Model()
#     all_personal_m = Model()
#
#     rf4_m.load_state_dict(torch.load('E:/deeplearning/model/1994'))
#     rf5_m.load_state_dict(torch.load('E:/deeplearning/model/1995'))
#     rf6_m.load_state_dict(torch.load('E:/deeplearning/model/1996'))
#     rfall_m.load_state_dict(torch.load('E:/deeplearning/model/all'))
#     all_personal_m.load_state_dict(torch.load('E:/deeplearning/model/all_personal'))
#
#     params_list = [rf4_m.state_dict(), rf5_m.state_dict(), rf6_m.state_dict()]
#
#     test_data = Test_Data(datas)
#     dataloader = DataLoader(test_data, batch_size=len(test_data), shuffle=True, num_workers=1)
#
#     loss_1994_1994, accuracy_1994_1994 = test(rf4_m, dataloader)
#     loss_1995_1995, accuracy_1995_1995 = test(rf5_m, dataloader)
#     loss_1996_1996, accuracy_1996_1996 = test(rf6_m, dataloader)
#     loss_rfall_m, accuracy_rfall_m = test(rfall_m, dataloader)
#     loss_all_personal_m, accuracy_all_personal_m = test(all_personal_m, dataloader)
#
#     print(f"loss_1994_1994 - Loss: {loss_1994_1994}, accuracy_1994_1994: {accuracy_1994_1994}")
#     print(f"loss_1995_1995 - Loss: {loss_1995_1995}, accuracy_1995_1995: {accuracy_1995_1995}")
#     print(f"loss_1996_1996 - Loss: {loss_1996_1996}, accuracy_1996_1996: {accuracy_1996_1996}")
#     print(f"loss_rfall_m - Loss: {loss_rfall_m}, accuracy_rfall_m: {accuracy_rfall_m}")
#     print(f"loss_all_personal_m - Loss: {loss_all_personal_m}, accuracy_all_personal_m: {accuracy_all_personal_m}")
#
#     loss, accuracy = test(all_personal_m, dataloader)
#
#     print(f'global_step:{global_steps}\tloss:{loss.item()}\taccuracy:{accuracy}')
#     with open('result.txt', 'a') as f:
#         f.writelines(f'{global_steps}\t{loss.item()}\t{accuracy}\n')
