import numpy as np

def MSE(y_pred, y_true):
    loss = np.sum((y_true - y_pred) ** 2) / len(y_true)

    return loss

if __name__ == "__main__":
    y_tr = np.array([0, 1, 1, 0])
    y_pr = np.array([0.5, 0.5, 0.5, 0.5])
    loss_ = MSE(y_pr, y_tr)
    print(loss_)