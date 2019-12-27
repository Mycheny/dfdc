import numpy as np


def loss(y_, y):
    loss = y * np.log(y_) + (1 - y) * np.log(1 - y_)
    loss = -np.mean(loss)
    return loss


if __name__ == '__main__':
    ok = 200
    ok = 400-ok
    y = np.zeros((400))+0.000000000000001
    y_ = np.zeros((400))+0.000000000000001
    y_[:ok] = np.ones((ok))-0.000000000000001
    print(loss(y_, y))
