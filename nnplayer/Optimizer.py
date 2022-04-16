class Optimizer:
    def __init__(self, paramsList: list[dict]):
        self.paramsList = paramsList

    def step(self):
        raise NotImplementedError


class SGD(Optimizer):
    def __init__(self, paramsList: list[dict], learning_rate=0.1, weight_decay=0):
        super().__init__(paramsList)
        self.lr = learning_rate
        self.wd = weight_decay

    def step(self):
        for params in reversed(self.paramsList):
            for paramName in params:
                params[paramName]['data'] -= params[paramName]['gradient'] * self.lr
