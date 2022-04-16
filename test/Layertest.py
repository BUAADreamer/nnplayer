import numpy as np

import nnplayer as npl
from test.dataset import getData1

X, y, X_test, y_test = getData1(0.8)
net = npl.Sequence(npl.Linear(6, 4), npl.Linear(4, 1))
epoch = 100000
loss = npl.MSELoss(net)
optimizer = npl.SGD(net.getParamsList())
batch_size = 2
for i in range(epoch):
    for j in range(0, len(X), batch_size):
        Y_predict = net.forward(X[j:j + batch_size])
        # print(Y_predict)
        l = loss.loss(Y_predict, y[j:j + batch_size])
        loss.backward()
        optimizer.step()
    Y_predict = net.forward(X)
    l = loss.loss(Y_predict, y)
    print(f'epoch{i + 1},{l.mean()}')
y_pred = net.forward(X)
l = loss.loss(y_pred, y)
print(l.mean())
# print(y_pred, y)
