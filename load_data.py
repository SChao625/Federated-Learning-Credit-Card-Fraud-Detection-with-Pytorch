from imblearn.over_sampling import SMOTE
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from sklearn.preprocessing import LabelEncoder
from torch import Tensor
from torch.utils.data import Dataset


class Datas:

    def __init__(self):
        super().__init__()
        df = pd.read_csv("carclaims.csv")
        le = LabelEncoder()
        cols = df.select_dtypes('O').columns
        df[cols] = df[cols].apply(le.fit_transform)
        df['Year'] = le.fit_transform(df.Year)

        df1 = df[df["AccidentArea"] == 1]
        # df1 = df[df["AccidentArea"] == 1]

        df1 = df1.drop(
            columns=['DayOfWeek', 'PolicyNumber', 'Year', 'Age', 'BasePolicy', 'Month', 'PolicyType', 'WeekOfMonth',
                     'MaritalStatus'])
        one_df1 = pd.get_dummies(df1, prefix_sep="_", columns=df1.columns)
        data1 = np.array(one_df1, dtype=np.float32)
        #print(data)
        x1 = data1[::, 0:-2:]
        y1 = data1[::, -2:-1:]
        smote = SMOTE()
        x1, y1 = smote.fit_resample(x1, y1)

        df = df.drop(
            columns=['DayOfWeek', 'PolicyNumber', 'Year', 'Age', 'BasePolicy', 'Month', 'PolicyType', 'WeekOfMonth',
                     'MaritalStatus'])
        one_df = pd.get_dummies(df, prefix_sep="_", columns=df.columns)
        data = np.array(one_df, dtype=np.float32)

        #print(data)
        x = data[::, 0:-2:]
        y = data[::, -2:-1:]
        smote = SMOTE()
        x, y = smote.fit_resample(x, y)

        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x1, y1, test_size=0.2)
        # self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y, test_size=0.2)


    def get_data(self):
        return self.x_train, self.x_test, self.y_train, self.y_test


class Train_Data(Dataset):
    def __init__(self, datas):
        super().__init__()
        self.x_train, x_test, self.y_train, y_test = datas.get_data()

    def __len__(self):
        return len(self.x_train)

    def __getitem__(self, item):
        feature = torch.zeros([1, 256])
        for index, data in enumerate(self.x_train[item]):
            if 2 * index < 256:
                feature[0][2 * index] = torch.tensor(data)
        return feature.reshape((1, 16, 16)), torch.tensor(np.expand_dims([self.y_train[item]], axis=0))

    def get_train(self):
        return self.x_train


class Test_Data(Dataset):
    def __init__(self, datas):
        super().__init__()
        x_train, self.x_test, y_train, self.y_test = datas.get_data()

    def __len__(self):
        return len(self.x_test)

    def __getitem__(self, item):
        feature = torch.zeros([1, 256])
        for index, data in enumerate(self.x_test[item]):
            if 2 * index < 256:
                feature[0][2 * index] = torch.tensor(data)
        return feature.reshape((1, 16, 16)), torch.tensor(np.expand_dims([self.y_test[item]], axis=0))

# if __name__ == "__main__":
#     datas = Datas()
#     train_data = Train_Data(datas)
#     # 将数据集分为 n 个部分
#     n = 5
#     data_len = len(train_data)
#     subset_len = data_len // n
#     # 创建 Subset 对象的列表
#     subsets = [Subset(train_data, list(range(i * subset_len, (i + 1) * subset_len))) for i in range(n)]
#     data = [subset for subset in subsets]
#     dataloader = DataLoader(data[0], batch_size=64, shuffle=True, num_workers=1)
#     for index, (x, y) in enumerate(dataloader):
#         print(index, x.shape[0])
