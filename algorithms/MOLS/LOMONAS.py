from algorithms.MOLS import Algorithm
from utils import check_valid
from utils import get_hashKey
from utils import Individual
import numpy as np
import time
from copy import deepcopy
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting


def sample(arch_history, problem):
    while True:
        X = problem.sample_a_compact_architecture()
        if problem.isValid(X):
            hashKey = get_hashKey(X, problem.name + '*')
            if hashKey not in arch_history:
                return X


def get_partial_neighbors(X_arch, arch_history, problem):
    problem_name = problem.name
    hashKey_arch = get_hashKey(X_arch, problem_name + '*')

    if hashKey_arch in arch_history:
        if len(arch_history[hashKey_arch]) == 0:
            return [], arch_history
        available_idx = arch_history[hashKey_arch]
        idx_replace = np.random.choice(available_idx)
        arch_history[hashKey_arch].remove(idx_replace)
    else:
        if problem_name == 'NASBench101':
            available_idx = list(range(len(X_arch)))
            available_idx.remove(0)
            available_idx.remove(21)
        elif problem_name in ['NASBench201', 'MacroNAS', 'NASBenchASR']:
            available_idx = list(range(len(X_arch)))
        else:
            raise ValueError()
        arch_history[hashKey_arch] = available_idx
        idx_replace = np.random.choice(arch_history[hashKey_arch])
        arch_history[hashKey_arch].remove(idx_replace)

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
    available_ops_at_idx_replace.remove(X_arch[idx_replace])

    list_X_neighbors = []
    for op in available_ops_at_idx_replace:
        X_neighbor = X_arch.copy()
        X_neighbor[idx_replace] = op
        list_X_neighbors.append(X_neighbor)
    return list_X_neighbors, arch_history


def get_all_neighbors(X_arch, arch_history, problem):
    problem_name = problem.name
    hashKey_arch = get_hashKey(X_arch, problem_name + '*')

    if hashKey_arch in arch_history:
        return [], arch_history
    list_X_neighbors = []

    if problem_name == 'NASBench101':
        available_idx = list(range(len(X_arch)))
        available_idx.remove(0)
        available_idx.remove(21)
    elif problem_name in ['NASBench201', 'MacroNAS', 'NASBenchASR']:
        available_idx = list(range(len(X_arch)))
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
        available_ops_at_idx_replace.remove(X_arch[idx_replace])
        for op in available_ops_at_idx_replace:
            X_neighbor = X_arch.copy()
            X_neighbor[idx_replace] = op
            list_X_neighbors.append(X_neighbor)
    arch_history[hashKey_arch] = list_X_neighbors
    return list_X_neighbors, arch_history


####################################### Get architectures for local search #############################################

## Get all architectures
def get_all_archs_for_local_search(X_set, F_set, NF):
    idx_fronts = NonDominatedSorting().do(np.array(F_set))
    selected_idx = np.zeros(len(F_set), dtype=bool)
    N = min(len(idx_fronts), NF)
    for i in range(N):
        for j in idx_fronts[i]:
            selected_idx[j] = True
    X_local_search_list = np.array(X_set)[selected_idx].tolist()
    return X_local_search_list


## Get potential architectures (knee and extreme ones)
def get_potential_archs_for_local_search(X_set, F_set, NF):
    X_list = []

    idx_fronts = NonDominatedSorting().do(np.array(F_set))
    N = min(len(idx_fronts), NF)
    for i in range(N):
        selected_idx = np.zeros(len(F_set), dtype=bool)
        selected_idx[idx_fronts[i]] = True

        X_front_i = np.array(X_set)[selected_idx]
        F_front_i = np.array(F_set)[selected_idx]

        potential_sols = seeking(X_front_i, F_front_i)
        potential_sols_list = np.array([info[1] for info in potential_sols])

        for X in potential_sols_list:
            X_list.append(X)
    return X_list


########################################## Multi-objective Local Search ################################################
## Get all neighbors (for creating the neighbors dictionary)
def _get_all_neighbors(problem, arch):
    problem_name = problem.name

    list_neighbors = []

    if problem_name == 'NASBench101':
        available_idx = list(range(len(arch)))
        available_idx.remove(0)
        available_idx.remove(21)
    elif problem_name in ['NASBench201', 'MacroNAS', 'NASBenchASR']:
        available_idx = list(range(len(arch)))
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
        available_ops_at_idx_replace.remove(arch[idx_replace])
        for op in available_ops_at_idx_replace:
            neighbor_X = arch.copy()
            neighbor_X[idx_replace] = op
            list_neighbors.append(neighbor_X)
    return list_neighbors

class LOMONAS(Algorithm):
    """
    LOMONAS - Local search algorithm for Multi-objective Neural Architecture Search
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Default: f0 -> performance metric; f1 -> efficiency metric
        self.f0, self.f1 = None, None
        self.NF = 3
        self.get_all_neighbors = False
        self.local_search_on_all_sols = True

    def _reset(self):
        pass

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
        self.nEvals += 1
        self.benchmark_time_algorithm_history.append(self.benchmark_time_algorithm_history[-1] + benchmark_time)
        self.indicator_time_history.append(self.indicator_time_history[-1] + indicator_time)
        self.evaluated_time_history.append(self.evaluated_time_history[-1] + benchmark_time + indicator_time)
        self.running_time_history.append(self.evaluated_time_history[-1] + self.executed_time_algorithm_history[-1])
        self.start_executed_time_algorithm = time.time()
        return performance_metric + efficiency_metric

    def get_list_neighbors(self, Q, arch_history, hashKey_SF, get_all_neighbors_flag=None):
        if get_all_neighbors_flag is None:
            get_all_neighbors_flag = self.get_all_neighbors

        _arch_history = arch_history
        list_X_neighbors = []
        list_hashKey_neighbors = []

        for i, arch in enumerate(Q):
            if get_all_neighbors_flag:
                tmp_list_X_neighbors, _arch_history = get_all_neighbors(arch, _arch_history, self.problem)
            else:
                tmp_list_X_neighbors, _arch_history = get_partial_neighbors(arch, _arch_history, self.problem)

            for X_neighbor in tmp_list_X_neighbors:
                if self.problem.isValid(X_neighbor):
                    hashKey_neighbor = get_hashKey(X_neighbor, problem_name=self.problem.name)
                    if check_valid(hashKey_neighbor,
                                   list_hashKey_Q=hashKey_SF,
                                   list_hashKey_neighbors=list_hashKey_neighbors):
                        list_X_neighbors.append(X_neighbor)
                        list_hashKey_neighbors.append(hashKey_neighbor)
        return list_X_neighbors, list_hashKey_neighbors, _arch_history

    def _solve(self):
        self.start_executed_time_algorithm = time.time()
        is_continue = True

        arch_history = {}

        # lines 3 - 4
        X_start = sample(arch_history, self.problem)
        hashKey_start = get_hashKey(X_start, self.problem.name)
        F_start = self.evaluate(X_start)

        while is_continue:
            start_arch = Individual()
            start_arch.set('X', X_start)
            start_arch.set('hashKey', hashKey_start)
            start_arch.set('F', F_start)
            self.E_Archive_search.update(start_arch, algorithm=self, problem_name=self.problem.name)

            # line 6
            X_SF = [start_arch.X]
            hashKey_SF = [start_arch.hashKey]
            F_SF = [start_arch.F]

            Q = [start_arch.X]  # line 7

            while True:
                list_X_neighbors, list_hashKey_neighbors, arch_history = self.get_list_neighbors(Q,
                                                                                                 arch_history,
                                                                                                 hashKey_SF)  # line 9
                list_F_neighbors = []

                if len(list_X_neighbors) == 0:  # line 10
                    # lines 11 - 15
                    for f in range(2, self.NF + 1):
                        if self.local_search_on_all_sols:
                            Q = get_all_archs_for_local_search(X_SF, F_SF, f)
                        else:
                            Q = get_potential_archs_for_local_search(X_SF, F_SF, f)

                        list_X_neighbors, list_hashKey_neighbors, arch_history = self.get_list_neighbors(Q,
                                                                                                         arch_history,
                                                                                                         hashKey_SF)
                        if len(list_X_neighbors) != 0:
                            break

                    # lines 16 - 21
                    if len(list_X_neighbors) == 0:
                        X_archive = self.E_Archive_search.X
                        while True:
                            idx = np.random.choice(len(X_archive))
                            selected_arch = X_archive[idx]
                            list_neighbors = []
                            tmp_list_neighbors = _get_all_neighbors(self.problem, selected_arch)

                            for X_neighbor in tmp_list_neighbors:
                                if self.problem.isValid(X_neighbor):
                                    hashKey_neighbor = get_hashKey(X_neighbor, self.problem.name + '*')
                                    if hashKey_neighbor not in arch_history:
                                        list_neighbors.append(X_neighbor)
                            if len(list_neighbors) != 0:
                                break

                        idx_selected_neighbor = np.random.choice(len(list_neighbors))
                        X_start = list_neighbors[idx_selected_neighbor]
                        hashKey_start = get_hashKey(X_start, self.problem.name)
                        F_start = self.evaluate(X_start)
                        break

                # lines 23
                for X_neighbor in list_X_neighbors:
                    hashKey_neighbor = get_hashKey(X_neighbor, problem_name=self.problem.name)

                    F_neighbor = self.evaluate(X_neighbor)
                    list_F_neighbors.append(F_neighbor)

                    neighbor_arch = Individual()
                    neighbor_arch.set('X', X_neighbor)
                    neighbor_arch.set('hashKey', hashKey_neighbor)
                    neighbor_arch.set('F', F_neighbor)
                    self.E_Archive_search.update(neighbor_arch, algorithm=self, problem_name=self.problem.name)

                # line 24
                X_P = X_SF + list_X_neighbors
                hashKey_P = hashKey_SF + list_hashKey_neighbors
                F_P = F_SF + list_F_neighbors

                idx_fronts = NonDominatedSorting().do(np.array(F_P))
                idx_selected = np.zeros(len(F_P), dtype=bool)
                N = min(len(idx_fronts), self.NF)
                for i in range(N):
                    for j in idx_fronts[i]:
                        idx_selected[j] = True

                X_SF = np.array(deepcopy(X_P))[idx_selected].tolist()
                hashKey_SF = np.array(deepcopy(hashKey_P))[idx_selected].tolist()
                F_SF = np.array(deepcopy(F_P))[idx_selected].tolist()

                # line 25
                if self.local_search_on_all_sols:
                    Q = X_SF
                else:
                    Q = get_potential_archs_for_local_search(X_SF, F_SF, 1)

                if self.nEvals >= self.problem.maxEvals:
                    is_continue = False
                    break
                if self.debug:
                    print(f'-> nEvals / maxEvals: {self.nEvals}/{self.problem.maxEvals}')

        self.finalize()
        results = {
            'Approximation Set': self.E_Archive_search.X,
            'Approximation Front': self.E_Archive_search.F,
            'Search Cost': self.running_time_history[-1]
        }
        return results

    def log_elitist_archive(self, **kwargs):
        pass

    def _finalize(self):
        pass

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
            position = check_above_or_below(considering_pt=non_dominated_front[i],
                                            remaining_pt_1=non_dominated_front[l],
                                            remaining_pt_2=non_dominated_front[h])
            if position == -1:
                angle_measure = calculate_angle_measure(considering_pt=non_dominated_front_norm[i],
                                                        neighbor_1=non_dominated_front_norm[l],
                                                        neighbor_2=non_dominated_front_norm[h])
                if angle_measure > 210:
                    potential_sols.append([i, non_dominated_set[i], 'knee'])

    return potential_sols


def check_above_or_below(considering_pt, remaining_pt_1, remaining_pt_2):
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


def calculate_angle_measure(considering_pt, neighbor_1, neighbor_2):
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
