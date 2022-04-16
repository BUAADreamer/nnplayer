import numpy as np
import nnplayer as npl

if __name__ == '__main__':
    net = npl.Sequence(npl.Linear(10, 6), npl.Sigmoid(), npl.Linear(6, 4), npl.Sigmoid())
    loss = npl.MSELoss(net)
    optimizer = npl.SGD(net.getParamsList())
    X = np.arange(40).reshape(4, 10)
    Y = np.arange(16).reshape(4, 4)
    epoch = 100
    for i in range(epoch):
        Y_predict = net.forward(X)
        l = loss.loss(Y_predict, Y)
        print(f'epoch{i + 1} loss:{l.mean()}')
        loss.backward()
        optimizer.step()
    l = loss.loss(net.forward(X), Y)
    print(l.mean())
