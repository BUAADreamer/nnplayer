import numpy as np


class Loss:
    def __init__(self, model):
        self.model = model

    def loss(self, X, Y):
        raise NotImplementedError


class MSELoss(Loss):
    def __init__(self, model):
        super().__init__(model)

    def loss(self, Y_predict, Y_ans):
        self.Y_predict = Y_predict
        self.Y_ans = Y_ans
        return 0.5 * np.sum((self.Y_ans - self.Y_predict) ** 2) / self.Y_ans.shape[0]

    def gradient(self, Y_predict, Y_ans):
        return (Y_ans - Y_predict) / Y_ans.shape[0]

    def backward(self):
        self.model.backward(self.gradient(self.Y_predict, self.Y_ans))
