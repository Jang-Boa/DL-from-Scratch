"""곱셉계층 구현 예제 [https://velog.io/@ksj5738/%EC%97%AD%EC%A0%84%ED%8C%8C-%EA%B5%AC%ED%98%84%ED%95%98%EA%B8%B0   ]
입력이 x, y일 때, 
- forward(순전파): x*y
- backward(역전파): dx = 미분값 * y, dy = 미분값 * x
- forward 할 때 들어왔던 값들ㅇ르 저장하고 있어야 한다.
"""
class MultipleLayer:
    # 딥러닝 레이어의 초기화는 레이어에서 사용할 옵션이나 변수를 미리 준비
    def __init__(self):
        self.x = None
        self.y = None

    # 곱셈레이어틔 순전파에서 역전파에 필요한 변수를 저장
    def forward(self, x, y):
        self.x = x
        self.y = y
        out = x * y
        return out

    # forward 할 때 저장해 놓았던 x, y를 각각 반대 방향으로 미분값과 곱해서 리턴
    def backward(self, dout):
        dx = dout * self.y
        dy = dout * self.x
        return dx, dy

if __name__ == '__main__':
    apple = 100
    apple_num = 2
    tax = 1.1

    mul_apple_layer = MultipleLayer()
    mul_tax_layer = MultipleLayer()
    
    apple_price = mul_apple_layer.forward(apple, apple_num)
    price = mul_tax_layer.forward(apple_price, tax)

    print(price)

    dprice = 1
    dapple_price, dtax = mul_tax_layer.backward(dprice)
    dapple, dapple_num = mul_apple_layer.backward(dapple_price)

    print(dapple, dapple_num, dtax)