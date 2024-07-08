import numpy as np

"""
- 모든 계층은 forward()와 backward() Method를 가진다.
> forward(): 순전파
> backward(): 역전파
- 모든 계층은 인스턴스 변수인 params와 grads를 가진다.
> params: 가중치와 편향과 같은 매개변수을 담는 리스트  
> grads: params에 저장된 각 매개변수에 대응하여, 해당 매개변수의 기울기를 보관하는 리스트

"""

class Sigmoid:
    def __init__(self) -> None:
        self.params = []

    def forward(self, x):
        return 1 / (1 + np.exp(-x))
    
class Affine:
    def __init__(self, W, b):
        self.params = [W, b]

    def forward(self, x):
        W, b = self.params
        out = np.matmul(x, W) + b
        return out

class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size) -> None:
        I, H, O = input_size, hidden_size, output_size

        # 가중치와 편향 초기화
        W1 = np.random.randn(I, H)
        b1 = np.random.randn(H)
        W2 = np.random.randn(H, O)
        b2 = np.random.randn(O)

        # Layer
        self.layers = [
            Affine(W1, b1), 
            Sigmoid(), 
            Affine(W2, b2)
        ]

        # 모든 가중치를 리스트로 모은다. -> 학습해야 할 가중치 매개변수들을 params 리스트에 저장
        self.params = []
        for layer in self.layers:
            self.params += layer.params
        
    def predict(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x
    
x = np.random.randn(10, 2)
model = TwoLayerNet(2, 7, 3)
# print(model)
s = model.predict(x)
print(s.shape)
