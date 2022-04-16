# Introduction

This repo is a Nerual Network Framework for learning Deep-Learning and Machine-Learning.

This framework refers to PyTorch's Design and Usage.

## Usage

```python
import nnplayer as npl
import numpy as np

# build net
net = npl.Sequence(npl.Linear(10, 6), npl.Sigmoid(), npl.Linear(6, 4), npl.Sigmoid())
# calculate loss 
loss = npl.MSELoss(net)
# optimizer
optimizer = npl.SGD(net.getParamsList())
# build the dataset
X = np.arange(40).reshape(4, 10)
Y = np.arange(16).reshape(4, 4)
# train
epoch = 100
for i in range(epoch):
    Y_predict = net.forward(X)
    l = loss.loss(Y_predict, Y)
    print(f'epoch{i + 1} loss:{l.mean()}')
    loss.backward()
    optimizer.step()
l = loss.loss(net.forward(X), Y)
print(l.mean())
```