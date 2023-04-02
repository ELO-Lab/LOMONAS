import numpy as np
import matplotlib.pyplot as plt

def visualize_archive(AF, c='tab:blue', xlabel=None, ylabel=None, marker=None, label=None, path=None, fig_name=None):
    AF_ = np.array(AF)
    AF_[:, 0], AF_[:, 1] = AF_[:, 1], AF_[:, 0].copy()
    AF_ = np.unique(AF_, axis=0)
    X = AF_[:, 0]
    Y = AF_[:, 1]

    X_, Y_ = [], []
    for i in range(len(X)):
        X_.append(X[i])
        Y_.append(Y[i])
        if i < len(X) - 1:
            X_.append(X[i + 1])
            Y_.append(Y[i])

    plt.plot(X_, Y_, '--', c=c)

    plt.scatter(X, Y, label=label, edgecolor=c, facecolor='none', marker=marker)
    plt.legend(loc='best')
    plt.grid(linestyle='--')
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.title('Approximation Front')
    if fig_name is None:
        plt.savefig(f'{path}/approximation_front.jpg', bbox_inches='tight', pad_inches=0.1, dpi=300)
    else:
        plt.savefig(f'{path}/{fig_name}.jpg', bbox_inches='tight', pad_inches=0.1, dpi=300)
    plt.clf()