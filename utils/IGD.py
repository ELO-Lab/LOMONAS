import numpy as np

def calculate_Euclidean_distance(x1, x2):
    euclidean_distance = np.sqrt(np.sum((x1 - x2) ** 2))
    return euclidean_distance


def calculate_IGD_value(pareto_optimal_front, approximation_front):
    approximation_front = np.unique(approximation_front, axis=0)
    d = 0
    for s in pareto_optimal_front:
        d += min([calculate_Euclidean_distance(s, s_) for s_ in approximation_front])
    return round(d / len(pareto_optimal_front), 6)
