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

def visualize_IGD_value_and_nEvals(nEvals_history, IGD_history, path_results, fig_name=None):
    """
    - This function is used to visualize 'IGD_values' and 'nEvals' at the end of the search.
    """
    plt.xscale('log')
    plt.xlabel('#Evals')
    plt.ylabel('IGD value')
    plt.grid('--')
    plt.step(nEvals_history, IGD_history, where='post')
    if fig_name is None:
        plt.savefig(f'{path_results}/#Evals-IGD.jpg', bbox_inches='tight', pad_inches=0.1, dpi=300)
    else:
        plt.savefig(f'{path_results}/{fig_name}', bbox_inches='tight', pad_inches=0.1, dpi=300)
    plt.clf()

def visualize_HV_value_and_nEvals(nEvals_history, HV_history, path_results, fig_name=None):
    """
    - This function is used to visualize 'HV_values' and 'nEvals' at the end of the search.
    """
    plt.xscale('log')
    plt.xlabel('#Evals')
    plt.ylabel('HV value')
    plt.grid('--')
    plt.step(nEvals_history, HV_history, where='post')
    if fig_name is None:
        plt.savefig(f'{path_results}/#Evals-HV.jpg', bbox_inches='tight', pad_inches=0.1, dpi=300)
    else:
        plt.savefig(f'{path_results}/{fig_name}', bbox_inches='tight', pad_inches=0.1, dpi=300)
    plt.clf()


def visualize_Elitist_Archive_and_Pareto_Front(AF, POF, xlabel=None, ylabel=None, path=None, fig_name=None):
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

    plt.plot(X_, Y_, '--', c='tab:blue')

    plt.scatter(X, Y, edgecolor='tab:blue', facecolor='none', marker='o', label='Approximation Front')

    POF_ = np.array(POF)
    POF_[:, 0], POF_[:, 1] = POF_[:, 1], POF_[:, 0].copy()
    POF_ = np.unique(POF_, axis=0)
    X = POF_[:, 0]
    Y = POF_[:, 1]

    X_, Y_ = [], []
    for i in range(len(X)):
        X_.append(X[i])
        Y_.append(Y[i])
        if i < len(X) - 1:
            X_.append(X[i + 1])
            Y_.append(Y[i])

    plt.plot(X_, Y_, '--', c='tab:red')

    plt.scatter(X, Y, edgecolor='tab:red', facecolor='none', marker='s', label='Pareto-optimal Front')

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