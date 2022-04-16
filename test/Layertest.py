import numpy as np

import nnplayer as npl
from test.dataset import getData1

X, y, X_test, y_test = getData1(0.8)
print(X.shape, y.shape)
net = npl.Sequence(npl.Linear(6, 3), npl.Sigmoid(), npl.Linear(3, 1), npl.Sigmoid())
epoch = 10000
loss = npl.MSELoss(net)
optimizer = npl.SGD(net.getParamsList())
for i in range(epoch):
    Y_predict = net.forward(X)
    l = loss.loss(Y_predict, y)
    print(f'epoch{i + 1} loss:{l.mean()}')
    loss.backward()
    optimizer.step()
y_pred = net.forward(X)
l = loss.loss(y_pred, y)
print(l.mean())
print(y_pred, y)
