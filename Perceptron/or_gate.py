import numpy as np

# Perceptron: w = (1, 1)^T, b = 0.5
# y = x1 + x2 - 0.5

def OR(x1, x2):
    x = np.array([x1, x2])
    w = np.array([1, 1]).T
    b = -0.5
    y = np.sum(x*w) + b
    if y <= 0:
        return -1
    else:
        return 1

if __name__ == '__main__':
    X = np.array([[0,0], [1,0], [0,1], [1,1]])
    # Y = np.array([-1,1,1,1])
    for x in X:
        print(x)
        y = OR(x[0], x[1])
        print(y)
    