import numpy as np
import matplotlib.pyplot as plt
import pdb
import os

save_data_dir = os.path.join(os.path.dirname(__file__), "iter_result")
os.makedirs(save_data_dir, exist_ok=True)

def draw(ax, *args):
    X, y, w1, w2, b = args
    # draw
    for i, x in enumerate(X):
        color = 'green' if y[i] == 1 else 'red'
        ax.scatter(x[0], x[1], c=color)
    # line
    if w1 == 0 and w2 == 0:
        return
    elif w1 == 0:
        n1, n2 = [-5, -b / w2], [5, -b / w2]
    elif w2 == 0:
        n1, n2 = [-b / w1, -5], [-b / w2, 5]
    n1 = [-b / w1, 0]
    n2 = [0, -b / w2]
    x, y = zip(n1, n2)
    ax.plot(x, y)

def main():
    X = [
        [3, 3],
        [4, 3],
        [1, 1]
    ]
    y = [1, 1, -1]

    N = 5
    w1, w2 = 0, 0
    b = 0
    for _ in range(N):
        for i in range(len(X)):
            if y[i] * (w1 * X[i][0] + w2 * X[i][1] + b) > 0:
                # corrext
                continue

            # update w
            w1 = w1 + y[i] * X[i][0]
            w2 = w2 + y[i] * X[i][1]
            b = b + y[i]

        # loss
        loss = 0
        for i in range(len(X)):
            loss += min(y[i] * (w1 * X[i][0] + w2 * X[i][1] + b), 0)
        print("Iterator {}, loss={}".format(_, loss))
        print("function: {}x1+{}x2+{}=0".format(w1, w2, b))

        # draw
        fig, axs = plt.subplots()
        draw(axs, X, y, w1, w2, b)
        save_path = os.path.join(save_data_dir, "iterator_{}.png".format(_))
        fig.savefig(save_path)

main()