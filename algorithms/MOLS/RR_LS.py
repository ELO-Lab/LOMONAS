"""
Source code for: Random Restart Local Search (RR-LS)
Coding from scratch basing on the idea from:
+ Paper: Local Search is a Remarkably Strong Baseline for Neural Architecture Search
+ Source Code: https://github.com/tdenottelander/MacroNAS
"""

from algorithms.MOLS import LOMONAS
from utils import Individual
import numpy as np
import time

def sample(problem):
    while True:
        x = problem.sample_a_compact_architecture()
        if problem.isValid(x):
            return x

def get_available_idx(x, problem):
    problem_name = problem.name
    if problem_name == 'NASBench101':
        available_idx = list(range(len(x)))
        available_idx.remove(0)
        available_idx.remove(21)
    elif problem_name in ['NASBench201', 'MacroNAS', 'NASBenchASR']:
        available_idx = list(range(len(x)))
    else:
        raise ValueError()
    return available_idx

def get_available_ops(x, idx, problem):
    problem_name = problem.name

    if problem_name == 'NASBench101':
        if idx in problem.IDX_OPS:
            available_ops = problem.OPS.copy()
        else:
            available_ops = problem.EDGES.copy()
    elif problem_name == 'NASBenchASR':
        if idx in problem.IDX_MAIN_OPS:
            available_ops = problem.MAIN_OPS.copy()
        else:
            available_ops = problem.SKIP_OPS.copy()
    elif problem_name in ['NASBench201', 'MacroNAS']:
        available_ops = problem.available_ops.copy()
    else:
        raise ValueError()
    available_ops_at_idx_replace = available_ops.copy()
    available_ops_at_idx_replace.remove(x[idx])
    return available_ops_at_idx_replace


########################################## Multi-objective Local Search ################################################
class RR_LS(LOMONAS):
    """
    RR_LS (loop)
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Default: f0 -> performance metric; f1 -> efficiency metric
        self.loop = False

    def scalarize_fitness(self, f, scalar_coff):
        f_0 = f[0]
        f_1 = self.problem.normalize_efficiency_metric(f[1])
        return f_0 * scalar_coff + f_1 * (1 - scalar_coff)

    def dominates(self, f_indThis, f_indOther, scalarization):
        scalar_f_this = self.scalarize_fitness(f_indThis, scalarization)
        scalar_f_other = self.scalarize_fitness(f_indOther, scalarization)
        return scalar_f_this < scalar_f_other

    def _solve(self):
        self.start_executed_time_algorithm = time.time()

        ls_archive = []
        while self.n_eval < self.problem.maxEvals:
            # 1. Initialize a random architecture
            x = sample(self.problem)
            f = self.evaluate(x)

            # Update Archive
            arch = Individual(x=x, f=f)
            self.E_Archive_search.update(arch, algorithm=self, problem_name=self.problem.name)

            best_x = arch.X.copy()
            best_f = arch.F.copy()

            change = True
            while change:
                change = False
                # 2. Sample a scalarization coefficient α ∼ U(0, 1);
                scalar_coef = np.random.uniform()

                # 3. Consider all variables in random order
                idx_list = np.random.permutation(get_available_idx(best_x, self.problem))

                for i in idx_list:
                    x_copy = best_x.copy()
                    ops_list = get_available_ops(x_copy, i, self.problem)
                    for op in ops_list:
                        x_copy[i] = op
                        if self.problem.isValid(x_copy):
                            f = self.evaluate(x_copy)

                            # Update Archive
                            arch = Individual(x=x_copy.copy(), f=f.copy())
                            self.E_Archive_search.update(arch, algorithm=self, problem_name=self.problem.name)

                            # 3.1. For each variable xi, evaluate the net obtained by setting xi to each option in Ωi.
                            # Keep the best according to α × f1 + (1 − α) × f2
                            if self.dominates(f, best_f, scalar_coef):
                                best_x = x_copy.copy()
                                best_f = f.copy()
                                if self.loop:
                                    change = True
                idv = {
                    'x': best_x,
                    'f': best_f
                }
                ls_archive.append(idv)
                if self.debug:
                    content = [
                        self.n_eval,
                        self.IGD_search_history[-1], self.IGDp_search_history[-1], self.HV_search_history[-1],
                        self.IGD_search_history[-1], self.IGDp_evaluate_history[-1], self.HV_evaluate_history[-1]
                    ]
                    print("-" * 104)
                    print(
                        "\033[92m{:<10}\033[00m | \033[96m{:^20.6f}\033[00m | \033[96m{:^20.6f}\033[00m | \033[96m{:^20.6f}\033[00m | \033[93m{:^20.6f}\033[00m | \033[93m{:^20.6f}\033[00m | \033[93m{:^20.6f}\033[00m |".format(
                            *content))

        self.finalize()
        results = {
            'Local Search Archive': ls_archive,
            'Approximation Set': self.E_Archive_search.X,
            'Approximation Front': self.E_Archive_search.F,
            'Search Cost': self.running_time_history[-1]
        }
        return results
