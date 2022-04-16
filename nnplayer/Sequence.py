from nnplayer.Model import Model


class Sequence(Model):
    def __init__(self, *layers):
        super().__init__()
        self.layers = layers

    def forward(self, X):
        output = X
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def backward(self, G):
        for layer in reversed(self.layers):
            G = layer.backward(G)
        return G

    def getParamsList(self):
        res = []
        for layer in self.layers:
            res.append(layer.params)
        return res
