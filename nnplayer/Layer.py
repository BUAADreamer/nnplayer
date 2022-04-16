from nnplayer import Init
import numpy as np
import copy


class Layer:
    def __init__(self):
        self.is_training = True
        self.params = {}

    def forward(self, X):
        raise NotImplementedError

    def backward(self, G):
        raise NotImplementedError


class Linear(Layer):
    def __init__(self, *shape):
        super().__init__()
        self.params['w'] = {}
        self.params['b'] = {}
        self.params['w']['data'] = Init.normal(shape)
        if len(shape) >= 2:
            self.params['b']['data'] = Init.normal(shape[1])
        self.params['w']['gradient'] = []
        self.params['b']['gradient'] = []

    def forward(self, X):
        self.X = X
        self.Y = np.matmul(self.X, self.params['w']['data']) + self.params['b']['data']
        return self.Y

    def backward(self, G):
        self.params['w']['gradient'] = np.matmul(self.X.T, G)
        self.params['b']['gradient'] = np.sum(G, axis=0)
        # print(G.shape, self.params['w']['data'].shape, self.X.shape)
        return np.matmul(G, self.params['w']['data'].T)


class Activate(Layer):
    def forward(self, X):
        self.X = X
        self.Y = self.function(X)
        return self.Y

    def backward(self, G):
        # print(self.X.shape, G.shape)
        return self.derivative(self.X) * G

    def function(self, X):
        raise NotImplementedError

    def derivative(self, X):
        raise NotImplementedError


class ReLU(Activate):

    def function(self, X):
        return np.maximum(X, 0)

    def derivative(self, X):
        return X > 0


class Sigmoid(Activate):
    def function(self, X):
        return 1 / (1 + np.exp(-X))

    def derivative(self, X):
        f = self.function(X)
        return f * (1 - f)
