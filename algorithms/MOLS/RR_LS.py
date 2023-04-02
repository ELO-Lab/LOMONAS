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
        X = problem.sample_a_compact_architecture()
        if problem.isValid(X):
            return X

def get_available_idx(X, problem):
    problem_name = problem.name
    if problem_name == 'NASBench101':
        available_idx = list(range(len(X)))
        available_idx.remove(0)
        available_idx.remove(21)
    elif problem_name in ['NASBench201', 'MacroNAS', 'NASBenchASR']:
        available_idx = list(range(len(X)))
    else:
        raise ValueError()
    return available_idx

def get_available_ops(X, idx, problem):
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
    available_ops_at_idx_replace.remove(X[idx])
    return available_ops_at_idx_replace


########################################## Multi-objective Local Search ################################################
class RR_LS(LOMONAS):
    """
    RR_LS (loop)
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Default: f0 -> performance metric; f1 -> efficiency metric
        self.f0, self.f1 = None, None
        self.loop = False

    def _reset(self):
        pass

    def scalarize_fitness(self, F, scalar_coff):
        F_0 = F[0]
        F_1 = self.problem.normalize_efficiency_metric(F[1])
        return F_0 * scalar_coff + F_1 * (1 - scalar_coff)

    def dominates(self, F_indThis, F_indOther, scalarization):
        scalar_F_this = self.scalarize_fitness(F_indThis, scalarization)
        scalar_F_other = self.scalarize_fitness(F_indOther, scalarization)
        return scalar_F_this < scalar_F_other

    def _solve(self):
        self.start_executed_time_algorithm = time.time()

        ls_archive = []
        while self.nEvals < self.problem.maxEvals:
            # 1. Initialize a random architecture
            X = sample(self.problem)
            F = self.evaluate(X)

            # Update Archive
            arch = Individual()
            arch.set('X', X)
            arch.set('F', F)
            self.E_Archive_search.update(arch, algorithm=self, problem_name=self.problem.name)

            best_X = arch.X.copy()
            best_F = arch.F.copy()

            change = True
            while change:
                change = False
                # 2. Sample a scalarization coefficient α ∼ U(0, 1);
                scalar_coef = np.random.uniform()

                # 3. Consider all variables in random order
                idx_list = np.random.permutation(get_available_idx(best_X, self.problem))

                for i in idx_list:
                    X_copy = best_X.copy()
                    ops_list = get_available_ops(X_copy, i, self.problem)
                    for op in ops_list:
                        X_copy[i] = op
                        if self.problem.isValid(X_copy):
                            F = self.evaluate(X_copy)

                            # Update Archive
                            arch = Individual()
                            arch.set('X', X_copy.copy())
                            arch.set('F', F.copy())
                            self.E_Archive_search.update(arch, algorithm=self, problem_name=self.problem.name)

                            # 3.1. For each variable xi, evaluate the net obtained by setting xi to each option in Ωi.
                            # Keep the best according to α × f1 + (1 − α) × f2
                            if self.dominates(F, best_F, scalar_coef):
                                best_X = X_copy.copy()
                                best_F = F.copy()
                                if self.loop:
                                    change = True
                idv = {
                    'X': best_X,
                    'F': best_F
                }
                ls_archive.append(idv)
                if self.debug:
                    print(f'-------------------------------------------------------------')
                    print(f'-> nEvals / maxEvals: {self.nEvals}/{self.problem.maxEvals}')

        self.finalize()
        results = {
            'Local Search Archive': ls_archive,
            'Approximation Set': self.E_Archive_search.X,
            'Approximation Front': self.E_Archive_search.F,
            'Search Cost': self.running_time_history[-1]
        }
        return results
