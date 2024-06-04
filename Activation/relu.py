# ReLU(x) = max(0, x)

class ReLU:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.x = (x <= 0)
        out = x.copy()
        out[self.mask] = 0
        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout
        return dx