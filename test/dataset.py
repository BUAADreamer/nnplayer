import numpy as np
import pandas as pd

num = 17


def getData1(train_ratio=0.8, m=6, n=6):
    df = pd.read_excel("data1.xlsx", index_col=0)
    train_num = num * train_ratio
    test_num = num - train_num
    train_data = df.loc[:train_num - 1, :]
    test_data = df.loc[train_num:, :]
    X = [df.iloc[i, :m].tolist() for i in train_data.index]
    y = [df.iloc[i, n].tolist() for i in train_data.index]
    X_test = [df.iloc[i, :m].tolist() for i in test_data.index]
    y_test = [df.iloc[i, n].tolist() for i in test_data.index]
    X = np.array(X)
    y = np.array(y).reshape(13, 1)
    X_test = np.array(X_test)
    y_test = np.array(y_test).reshape(3, 1)
    return X, y, X_test, y_test


def judgeTest(y_pred, y_test):
    rightnum = 0
    for i in range(len(y_pred)):
        ans = 0
        my = 0
        if y_test[i] == 1:
            ans = 1
            print("答案:好瓜")
        else:
            print("答案:坏瓜")
        if y_pred[i] >= 0.5:
            print("预测:好瓜")
            my = 1
        else:
            print("预测:坏瓜")
            my = 0
        if my == ans:
            rightnum += 1
    print("正确率:", "%d%%" % (rightnum / len(y_test) * 100))
