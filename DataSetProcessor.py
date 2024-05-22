import numpy as np
import pandas as pd
import torch
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.nn import functional as F
from tqdm import tqdm

from model import Model


class DataSetProcessor:
    def __init__(self, data, category_col, label_col, test_size=0.2):
        self.data = data
        self.category_col = category_col
        self.label_col = label_col
        self.test_size = test_size
        self.subsets = {}
        self.process_data()

    def process_data(self):
        # 对整个数据集进行one-hot编码，获取所有可能的字段
        grouped = self.data.groupby(self.category_col)
        self.data = self.data.drop(self.category_col, axis=1)

        one_hot_df = pd.get_dummies(self.data,prefix_sep="_", columns=self.data.columns)

        print(one_hot_df)
        one_hot_df = one_hot_df.drop(columns=['FraudFound_1'])
        self.one_hot_columns = one_hot_df.columns
        # 按照类别列分组

        for category, group in grouped:
            one_hot_group = self.one_hot_encode_and_align(group)
            X_resampled, y_resampled = self.smote_and_split(one_hot_group)
            X_train, X_test, y_train, y_test = self.split_train_test(X_resampled, y_resampled)
            self.subsets[category] = {
                'X_train': X_train.values,
                'X_test': X_test.values,
                'y_train': y_train.values,
                'y_test': y_test.values
            }

    def one_hot_encode_and_align(self, subset):
        one_hot_subset = pd.get_dummies(subset,prefix_sep="_", columns=subset.columns)
        # 确保子集的one-hot编码字段与原数据集一致，不存在的字段补0
        for col in self.one_hot_columns:
            if col not in one_hot_subset:
                one_hot_subset[col] = 0
        return one_hot_subset[self.one_hot_columns]

    def smote_and_split(self, subset):
        smote = SMOTE()
        X = subset.drop(columns=[self.label_col])
        y = subset[self.label_col]
        X_resampled, y_resampled = smote.fit_resample(X, y)
        return X_resampled, y_resampled

    def split_train_test(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, random_state=42)
        return X_train, X_test, y_train, y_test



def pre_data():
    df = pd.read_csv("F:/project/Federated-Learning-Credit-Card-Fraud-Detection-with-Pytorch/carclaims.csv")
    le = LabelEncoder()
    cols = df.select_dtypes('O').columns
    df[cols] = df[cols].apply(le.fit_transform)
    df['Year'] = le.fit_transform(df.Year)
    df = df.drop(
        columns=['DayOfWeek', 'PolicyNumber', 'Year', 'Age', 'BasePolicy', 'Month', 'PolicyType', 'WeekOfMonth',
                 'MaritalStatus',])

    return df


model = Model()
rounds = 10
local_epochs = 1
batch_size = 32

# 定义自定义数据集类
class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, item):
        feature = torch.zeros([1, 256])
        for index, data in enumerate(self.X[item]):
            if 2 * index < 256:
                feature[0][2 * index] = torch.tensor(data)
        return feature.reshape((1, 16, 16)), torch.tensor(np.expand_dims([self.y[item]], axis=0))

def local_train(category,X_train, y_train, local_model):
    optimizer = optim.SGD(local_model.parameters(), lr=0.001)
    lossfn = F.binary_cross_entropy_with_logits
    dataset = CustomDataset(X_train, y_train)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    total_steps = len(dataloader) * local_epochs
    progress_bar = tqdm(total=total_steps, position=0, leave=True)  # 设置 leave=True 保留进度条在终端
    for epoch in range(local_epochs):
        for step, (x, y) in enumerate(dataloader):
            optimizer.zero_grad()
            outputs = local_model(x)
            y = y.float()  # 确保目标标签是浮点数
            loss = lossfn(outputs, y)
            loss.backward()
            optimizer.step()
            # 更新进度条
            progress_bar.update(1)
            progress_bar.set_postfix({'id': category, 'Epoch': epoch, 'Step': step, 'Loss': loss.item()})
    progress_bar.close()

if __name__ == '__main__':

    # 创建DataFrame
    df = pre_data()

    # 使用类处理数据集
    processor = DataSetProcessor(df, category_col='RepNumber', label_col='FraudFound_0')

    for round in range(rounds):
        print(f'Round {round + 1}/{rounds}')
        for category, data in processor.subsets.items():
            local_train(category,data['X_train'],data['y_train'], model)

        server.average(clients)



        if round % 1 == 0:
            loss, accuracy = server.test()
            print(f'global_step:{round}\tloss:{loss.item()}\taccuracy:{accuracy}')
            with open('result.txt', 'a') as f:
                f.writelines(f'{round}\t{loss.item()}\t{accuracy}\n')