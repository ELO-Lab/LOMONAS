from algorithms.MOLS import Algorithm
from utils import check_valid
from utils import get_hashKey
from utils import Individual, ElitistArchive
import numpy as np
import pickle as p
import time
from copy import deepcopy
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from utils import visualize_IGD_value_and_nEvals, visualize_HV_value_and_nEvals

class LOMONAS(Algorithm):
    """
    LOMONAS - Local search algorithm for Multi-objective Neural Architecture Search
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Default: f0 -> performance metric; f1 -> efficiency metric
        self.f0, self.f1 = None, None
        self.NF = 3
        self.check_all_neighbors = False
        self.neighborhood_check_on_all_sols = True

        self.IGD_search_history = []
        self.IGDp_search_history = []
        self.HV_search_history = []

    def _reset(self):
        self.IGD_search_history = []
        self.IGDp_search_history = []
        self.HV_search_history = []

    def _setup(self):
        attr_perf_metric = self.f0.split('_')
        self.epoch = int(attr_perf_metric[-1])

    def _evaluate(self, arch):
        self.finish_executed_time_algorithm = time.time()
        executed_time_algorithm = self.executed_time_algorithm_history[-1] + (
                self.finish_executed_time_algorithm - self.start_executed_time_algorithm)

        performance_metric, benchmark_time, indicator_time, evaluate_before = self.problem.get_val_performance_metric(
            arch,
            epoch=self.epoch)
        efficiency_metric = self.problem.get_efficiency_metric(arch=arch)

        self.executed_time_algorithm_history.append(executed_time_algorithm)
        self.n_eval += 1
        self.benchmark_time_algorithm_history.append(self.benchmark_time_algorithm_history[-1] + benchmark_time)
        self.indicator_time_history.append(self.indicator_time_history[-1] + indicator_time)
        self.evaluated_time_history.append(self.evaluated_time_history[-1] + benchmark_time + indicator_time)
        self.running_time_history.append(self.evaluated_time_history[-1] + self.executed_time_algorithm_history[-1])
        self.start_executed_time_algorithm = time.time()
        return performance_metric + efficiency_metric

    def _solve(self):
        self.start_executed_time_algorithm = time.time()
        is_continue = True

        H = {}

        # lines 3 - 4
        x_start = sample(H, self.problem)
        hashKey_start = get_hashKey(x_start, self.problem.name)
        f_start = self.evaluate(x_start)

        while is_continue:
            start_arch = Individual(x=x_start, hashKey=hashKey_start, f=f_start)
            self.E_Archive_search.update(start_arch, algorithm=self, problem_name=self.problem.name)

            X_S, hashKey_S, F_S = [start_arch.X], [start_arch.hashKey], [start_arch.F]  # line 6

            Q = [start_arch.X]  # line 7

            while True:
                X_N, hashKey_N, H = self.getNeighbors(Q, H, hashKey_S)  # line 9
                F_N = []

                if len(X_N) == 0:  # line 10
                    # lines 11 - 15
                    for fid in range(2, self.NF + 1):
                        if self.neighborhood_check_on_all_sols:
                            Q = getAll4NeighborhoodCheck(X_S, F_S, fid)
                        else:
                            Q = getPotential4NeighborhoodCheck(X_S, F_S, fid)

                        X_N, hashKey_N, H = self.getNeighbors(Q, H, hashKey_S)
                        if len(X_N) != 0:
                            break

                    # lines 16 - 21
                    if len(X_N) == 0:
                        X_archive = self.E_Archive_search.X
                        while True:
                            selected_arch = X_archive[np.random.choice(len(X_archive))]
                            X_N = []
                            tmp_N = _getAllNeighbors(self.problem, selected_arch)

                            for x in tmp_N:
                                if self.problem.isValid(x):
                                    hashKey = get_hashKey(x, self.problem.name + '*')
                                    if hashKey not in H:
                                        X_N.append(x)
                            if len(X_N) != 0:
                                break

                        x_start = X_N[np.random.choice(len(X_N))]
                        hashKey_start = get_hashKey(x_start, self.problem.name)
                        f_start = self.evaluate(x_start)
                        break

                # lines 23
                for x in X_N:
                    hashKey = get_hashKey(x, problem_name=self.problem.name)

                    f = self.evaluate(x)
                    F_N.append(f)

                    neighbor_arch = Individual(x=x, hashKey=hashKey, f=f)
                    self.E_Archive_search.update(neighbor_arch, algorithm=self, problem_name=self.problem.name)

                # line 24
                X_P = X_S + X_N
                hashKey_P = hashKey_S + hashKey_N
                F_P = F_S + F_N

                idx_fronts = NonDominatedSorting().do(np.array(F_P))
                idx_selected = np.zeros(len(F_P), dtype=bool)
                N = min(len(idx_fronts), self.NF)
                for i in range(N):
                    for j in idx_fronts[i]:
                        idx_selected[j] = True

                X_S = np.array(deepcopy(X_P))[idx_selected].tolist()
                hashKey_S = np.array(deepcopy(hashKey_P))[idx_selected].tolist()
                F_S = np.array(deepcopy(F_P))[idx_selected].tolist()

                # line 25
                if self.neighborhood_check_on_all_sols:
                    Q = X_S
                else:
                    Q = getPotential4NeighborhoodCheck(X_S, F_S, 1)

                if self.n_eval >= self.problem.max_eval:
                    is_continue = False
                    break
                if self.debug:
                    content = [
                        self.n_eval,
                        self.IGD_search_history[-1], self.IGDp_search_history[-1], self.HV_search_history[-1],
                        self.IGD_evaluate_history[-1], self.IGDp_evaluate_history[-1], self.HV_evaluate_history[-1]
                    ]
                    print("-" * 150)
                    print("\033[92m{:<10}\033[00m | \033[96m{:^20.6f}\033[00m | \033[96m{:^20.6f}\033[00m | \033[96m{:^20.6f}\033[00m | \033[93m{:^20.6f}\033[00m | \033[93m{:^20.6f}\033[00m | \033[93m{:^20.6f}\033[00m |".format(*content))

        self.finalize()
        results = {
            'Approximation Set': self.E_Archive_search.X,
            'Approximation Front': self.E_Archive_search.F,
            'Search Cost': self.running_time_history[-1]
        }
        return results

    ########################################################################################################## Utilities
    def getNeighbors(self, Q, H, hashKey_S, get_all_neighbors_flag=None):
        if get_all_neighbors_flag is None:
            get_all_neighbors_flag = self.check_all_neighbors

        clone_H = H
        X_N, hashKey_N = [], []

        for i, arch in enumerate(Q):
            if get_all_neighbors_flag:
                tmp_X_N, clone_H = getAllNeighbors(arch, clone_H, self.problem)
            else:
                tmp_X_N, clone_H = getPartialNeighbors(arch, clone_H, self.problem)

            for x in tmp_X_N:
                if self.problem.isValid(x):
                    hashKey = get_hashKey(x, problem_name=self.problem.name)
                    if check_valid(hashKey,
                                   list_hashKey_S=hashKey_S,
                                   list_hashKey_neighbors=hashKey_N):
                        X_N.append(x)
                        hashKey_N.append(hashKey)
        return X_N, hashKey_N, clone_H

    def log_elitist_archive(self, **kwargs):
        E_Archive_evaluate = ElitistArchive(log_each_change=False)

        self.nEvals_history.append(self.n_eval)

        EA_search = {
            'X': self.E_Archive_search.X.copy(),
            'hashKey': self.E_Archive_search.hashKey.copy(),
            'F': self.E_Archive_search.F.copy(),
        }
        self.E_Archive_search_history.append(EA_search)

        """  In practice, these below steps are unworkable """
        # Evaluation Step (to visualize the trend of IGD, do not affect the process of NAS search)
        dummy_idv = Individual()
        ## Evaluate each architecture in the Elitist Archive
        for x in EA_search['X']:
            test_error = self.problem.get_test_performance(arch=x)
            efficiency_metric = self.problem.get_efficiency_metric(arch=x)
            f = test_error + efficiency_metric
            dummy_idv.set('X', x)
            dummy_idv.set('F', f)
            E_Archive_evaluate.update(dummy_idv, problem_name=self.problem.name)

        ## Calculate the IGD indicator.
        ## In practice, these steps are unworkable because we do not have the Pareto-optimal front
        approximation_front = np.array(E_Archive_evaluate.F)
        approximation_front = np.unique(approximation_front, axis=0)

        IGD_value_evaluate = self.problem.calculate_IGD(approximation_front=approximation_front)
        IGDp_value_evaluate = self.problem.calculate_IGDp(approximation_front=approximation_front)
        HV_value_evaluate = self.problem.calculate_HV(approximation_front=approximation_front)
        self.IGD_evaluate_history.append(IGD_value_evaluate)
        self.IGDp_evaluate_history.append(IGDp_value_evaluate)
        self.HV_evaluate_history.append(HV_value_evaluate)

        approximation_front_val = np.array(EA_search['F'])
        approximation_front_val = np.unique(approximation_front_val, axis=0)

        IGD_value_search = self.problem.calculate_IGD_val(approximation_front=approximation_front_val)
        IGDp_value_search = self.problem.calculate_IGDp_val(approximation_front=approximation_front_val)
        HV_value_search = self.problem.calculate_HV(approximation_front=approximation_front_val)
        self.IGD_search_history.append(IGD_value_search)
        self.IGDp_search_history.append(IGDp_value_search)
        self.HV_search_history.append(HV_value_search)

        EA_evaluate = {
            'X': E_Archive_evaluate.X.copy(),
            'hashKey': E_Archive_evaluate.hashKey.copy(),
            'F': E_Archive_evaluate.F.copy()
        }
        self.E_Archive_evaluate_history.append(EA_evaluate)

    def _finalize(self):
        p.dump([self.nEvals_history, self.IGD_search_history],
               open(f'{self.path_results}/#Evals_and_IGD_search.p', 'wb'))
        p.dump([self.nEvals_history, self.IGDp_search_history],
               open(f'{self.path_results}/#Evals_and_IGDp_search.p', 'wb'))
        p.dump([self.nEvals_history, self.HV_search_history],
               open(f'{self.path_results}/#Evals_and_HV_search.p', 'wb'))

        visualize_IGD_value_and_nEvals(IGD_history=self.IGD_search_history,
                                       nEvals_history=self.nEvals_history,
                                       path_results=self.path_results,
                                       fig_name='/#Evals-IGD_search')

        visualize_IGD_value_and_nEvals(IGD_history=self.IGDp_search_history,
                                       nEvals_history=self.nEvals_history,
                                       path_results=self.path_results,
                                       ylabel='IGD+ value',
                                       fig_name='/#Evals-IGDp_search')

        visualize_HV_value_and_nEvals(HV_history=self.HV_search_history,
                                      nEvals_history=self.nEvals_history,
                                      path_results=self.path_results,
                                      fig_name='/#Evals-HV_search')

#####################################################################################
def seeking(X_list, F_list):
    non_dominated_set = X_list.copy()
    non_dominated_front = F_list.copy()

    sorted_idx = np.argsort(non_dominated_front[:, 0])

    non_dominated_set = non_dominated_set[sorted_idx]
    non_dominated_front = non_dominated_front[sorted_idx]

    non_dominated_front_norm = non_dominated_front.copy()

    min_f0 = np.min(non_dominated_front[:, 0])
    max_f0 = np.max(non_dominated_front[:, 0])

    min_f1 = np.min(non_dominated_front[:, 1])
    max_f1 = np.max(non_dominated_front[:, 1])

    non_dominated_front_norm[:, 0] = (non_dominated_front_norm[:, 0] - min_f0) / (max_f0 - min_f0)
    non_dominated_front_norm[:, 1] = (non_dominated_front_norm[:, 1] - min_f1) / (max_f1 - min_f1)

    potential_sols = [
        [0, non_dominated_set[0], 'best_f0']  # (idx (in full set), property)
    ]

    for i in range(len(non_dominated_front) - 1):
        if np.sum(np.abs(non_dominated_front[i] - non_dominated_front[i + 1])) != 0:
            break
        else:
            potential_sols.append([i + 1, non_dominated_set[i + 1], 'best_f0'])

    for i in range(len(non_dominated_front) - 1, -1, -1):
        if np.sum(np.abs(non_dominated_front[i] - non_dominated_front[i - 1])) != 0:
            break
        else:
            potential_sols.append([i - 1, non_dominated_set[i - 1], 'best_f1'])
    potential_sols.append([len(non_dominated_front) - 1, non_dominated_set[len(non_dominated_front) - 1], 'best_f1'])

    ## find the knee solutions
    start_idx = potential_sols[0][0]
    end_idx = potential_sols[-1][0]

    for i in range(len(potential_sols)):
        if potential_sols[i + 1][-1] == 'best_f1':
            break
        else:
            start_idx = potential_sols[i][0] + 1

    for i in range(len(potential_sols) - 1, -1, -1):
        if potential_sols[i - 1][-1] == 'best_f0':
            break
        else:
            end_idx = potential_sols[i][0] - 1

    for i in range(start_idx, end_idx + 1):
        l = None
        h = None
        for m in range(i - 1, -1, -1):
            if np.sum(np.abs(non_dominated_front[m] - non_dominated_front[i])) != 0:
                l = m
                break
        for m in range(i + 1, len(non_dominated_front), 1):
            if np.sum(np.abs(non_dominated_front[m] - non_dominated_front[i])) != 0:
                h = m
                break

        if (h is not None) and (l is not None):
            position = checkAboveOrBelow(considering_pt=non_dominated_front[i],
                                            remaining_pt_1=non_dominated_front[l],
                                            remaining_pt_2=non_dominated_front[h])
            if position == -1:
                angle_measure = calculateAngleMeasure(considering_pt=non_dominated_front_norm[i],
                                                        neighbor_1=non_dominated_front_norm[l],
                                                        neighbor_2=non_dominated_front_norm[h])
                if angle_measure > 210:
                    potential_sols.append([i, non_dominated_set[i], 'knee'])

    return potential_sols


def checkAboveOrBelow(considering_pt, remaining_pt_1, remaining_pt_2):
    """
    This function is used to check if the considering point is above or below
    the line connecting two remaining points.\n
    1: above\n
    -1: below
    """
    orthogonal_vector = remaining_pt_2 - remaining_pt_1
    line_connecting_pt1_and_pt2 = -orthogonal_vector[1] * (considering_pt[0] - remaining_pt_1[0]) \
                                  + orthogonal_vector[0] * (considering_pt[1] - remaining_pt_1[1])
    if line_connecting_pt1_and_pt2 > 0:
        return 1
    return -1


def calculateAngleMeasure(considering_pt, neighbor_1, neighbor_2):
    """
    This function is used to calculate the angle measure is created by the considering point
    and two its nearest neighbors
    """
    line_1 = neighbor_1 - considering_pt
    line_2 = neighbor_2 - considering_pt
    cosine_angle = (line_1[0] * line_2[0] + line_1[1] * line_2[1]) \
                   / (np.sqrt(np.sum(line_1 ** 2)) * np.sqrt(np.sum(line_2 ** 2)))
    if cosine_angle < -1:
        cosine_angle = -1
    if cosine_angle > 1:
        cosine_angle = 1
    angle = np.arccos(cosine_angle)
    return 360 - np.degrees(angle)

def sample(H, problem):
    while True:
        x = problem.sample_a_compact_architecture()
        if problem.isValid(x):
            hashKey = get_hashKey(x, problem.name + '*')
            if hashKey not in H:
                return x


def getPartialNeighbors(x, H, problem):
    problem_name = problem.name
    hashKey = get_hashKey(x, problem_name + '*')

    if hashKey in H:
        if len(H[hashKey]) == 0:
            return [], H
        available_idx = H[hashKey]
        idx_replace = np.random.choice(available_idx)
        H[hashKey].remove(idx_replace)
    else:
        if problem_name == 'NASBench101':
            available_idx = list(range(len(x)))
            available_idx.remove(0)
            available_idx.remove(21)
        elif problem_name in ['NASBench201', 'MacroNAS', 'NASBenchASR']:
            available_idx = list(range(len(x)))
        else:
            raise ValueError()
        H[hashKey] = available_idx
        idx_replace = np.random.choice(H[hashKey])
        H[hashKey].remove(idx_replace)

    if problem_name == 'NASBench101':
        if idx_replace in problem.IDX_OPS:
            available_ops = problem.OPS.copy()
        else:
            available_ops = problem.EDGES.copy()
    elif problem_name == 'NASBenchASR':
        if idx_replace in problem.IDX_MAIN_OPS:
            available_ops = problem.MAIN_OPS.copy()
        else:
            available_ops = problem.SKIP_OPS.copy()
    elif problem_name in ['NASBench201', 'MacroNAS']:
        available_ops = problem.available_ops.copy()
    else:
        raise ValueError()
    available_ops_at_idx_replace = available_ops.copy()
    available_ops_at_idx_replace.remove(x[idx_replace])

    X_N = []
    for op in available_ops_at_idx_replace:
        x_n = x.copy()
        x_n[idx_replace] = op
        X_N.append(x_n)
    return X_N, H


def getAllNeighbors(x, H, problem):
    problem_name = problem.name
    hashKey = get_hashKey(x, problem_name + '*')

    if hashKey in H:
        return [], H
    X_N = []

    if problem_name == 'NASBench101':
        available_idx = list(range(len(x)))
        available_idx.remove(0)
        available_idx.remove(21)
    elif problem_name in ['NASBench201', 'MacroNAS', 'NASBenchASR']:
        available_idx = list(range(len(x)))
    else:
        raise ValueError()

    for idx_replace in available_idx:
        if problem_name == 'NASBench101':
            if idx_replace in problem.IDX_OPS:
                available_ops = problem.OPS.copy()
            else:
                available_ops = problem.EDGES.copy()
        elif problem_name == 'NASBenchASR':
            if idx_replace in problem.IDX_MAIN_OPS:
                available_ops = problem.MAIN_OPS.copy()
            else:
                available_ops = problem.SKIP_OPS.copy()
        elif problem_name in ['NASBench201', 'MacroNAS']:
            available_ops = problem.available_ops.copy()
        else:
            raise ValueError()
        available_ops_at_idx_replace = available_ops.copy()
        available_ops_at_idx_replace.remove(x[idx_replace])
        for op in available_ops_at_idx_replace:
            x_n = x.copy()
            x_n[idx_replace] = op
            X_N.append(x_n)
    H[hashKey] = X_N
    return X_N, H


####################################### Get architectures for local search #############################################

## Get all architectures
def getAll4NeighborhoodCheck(X, F, NF):
    idx_fronts = NonDominatedSorting().do(np.array(F))
    selected_idx = np.zeros(len(F), dtype=bool)
    N = min(len(idx_fronts), NF)
    for i in range(N):
        for j in idx_fronts[i]:
            selected_idx[j] = True
    neighborhoodCheck_X = np.array(X)[selected_idx].tolist()
    return neighborhoodCheck_X


## Get potential architectures (knee and extreme ones)
def getPotential4NeighborhoodCheck(X, F, NF):
    neighborhoodCheck_X = []

    idx_fronts = NonDominatedSorting().do(np.array(F))
    N = min(len(idx_fronts), NF)
    for i in range(N):
        selected_idx = np.zeros(len(F), dtype=bool)
        selected_idx[idx_fronts[i]] = True

        X_front_i = np.array(X)[selected_idx]
        F_front_i = np.array(F)[selected_idx]

        potential_sols = seeking(X_front_i, F_front_i)
        potential_sols_list = np.array([info[1] for info in potential_sols])

        for x in potential_sols_list:
            neighborhoodCheck_X.append(x)
    return neighborhoodCheck_X

## Get all neighbors (for creating the neighbors dictionary)
def _getAllNeighbors(problem, x):
    problem_name = problem.name

    X_N = []

    if problem_name == 'NASBench101':
        available_idx = list(range(len(x)))
        available_idx.remove(0)
        available_idx.remove(21)
    elif problem_name in ['NASBench201', 'MacroNAS', 'NASBenchASR']:
        available_idx = list(range(len(x)))
    else:
        raise ValueError()

    for idx_replace in available_idx:
        if problem_name == 'NASBench101':
            if idx_replace in problem.IDX_OPS:
                available_ops = problem.OPS.copy()
            else:
                available_ops = problem.EDGES.copy()
        elif problem_name == 'NASBenchASR':
            if idx_replace in problem.IDX_MAIN_OPS:
                available_ops = problem.MAIN_OPS.copy()
            else:
                available_ops = problem.SKIP_OPS.copy()
        elif problem_name in ['NASBench201', 'MacroNAS']:
            available_ops = problem.available_ops.copy()
        else:
            raise ValueError()
        available_ops_at_idx_replace = available_ops.copy()
        available_ops_at_idx_replace.remove(x[idx_replace])
        for op in available_ops_at_idx_replace:
            x_n = x.copy()
            x_n[idx_replace] = op
            X_N.append(x_n)
    return X_N