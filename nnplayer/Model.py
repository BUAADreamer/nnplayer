class Model:
    def __init__(self):
        pass

    def forward(self, X):
        raise NotImplementedError

    def backward(self, G):
        raise NotImplementedError

    def getParamsList(self):
        raise NotImplementedError
