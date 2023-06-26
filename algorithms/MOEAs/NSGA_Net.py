"""
Modified from: https://github.com/msu-coinlab/pymoo
"""
import time
import numpy as np
import pickle as p

from algorithms.MOEAs import Algorithm
from utils import ElitistArchive, Individual
from utils import visualize_IGD_value_and_nEvals, visualize_HV_value_and_nEvals

def compare(idv_1, idv_2):
    rank_1, rank_2 = idv_1.get('rank'), idv_2.get('rank')
    if rank_1 < rank_2:
        return idv_1
    elif rank_1 > rank_2:
        return idv_2
    else:
        cd_1, cd_2 = idv_1.get('crowding'), idv_2.get('crowding')
        if cd_1 > cd_2:
            return idv_1
        elif cd_1 < cd_2:
            return idv_2
        else:
            return idv_1

def select_parents(pop):
    parents_pool = []
    for _ in range(2):
        index_pool = np.random.permutation(len(pop)).reshape((len(pop) // 2, 2))
        for idx in index_pool:
            competitor_1, competitor_2 = pop[idx[0]], pop[idx[1]]
            winner = compare(competitor_1, competitor_2)
            parents_pool.append(winner)
    return np.array(parents_pool)

class NSGA_Net(Algorithm):
    """
    NSGA-Net (validation error rate - efficiency metrics)
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Default: f0 -> performance metric; f1 -> efficiency metric
        self.f0, self.f1 = None, None

        self.IGD_search_history = []
        self.IGDp_search_history = []
        self.HV_search_history = []

    def _reset(self):
        self.IGD_search_history = []
        self.IGDp_search_history = []
        self.HV_search_history = []

    def _setup(self):
        self.sampling.nSamples = self.pop_size
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
        self.running_time_history.append(
            self.evaluated_time_history[-1] + self.executed_time_algorithm_history[-1])
        self.start_executed_time_algorithm = time.time()
        return performance_metric + efficiency_metric

    def _solve(self):
        self.start_executed_time_algorithm = time.time()
        self.initialize()
        self.pop = self.survival.do(self.pop, self.pop_size)

        self.do_each_gen()
        while self.n_eval < self.problem.maxEvals:
            self.n_gen += 1
            self.next(self.pop)
            self.do_each_gen()
        self.finalize()
        results = {
            'Approximation Set': self.E_Archive_search.X,
            'Approximation Front': self.E_Archive_search.F,
            'Search Cost': self.running_time_history[-1]
        }
        return results

    def _initialize(self):
        P = self.sampling.do(self.problem)
        for i in range(self.pop_size):
            F = self.evaluate(P[i].X)
            P[i].set('F', F)
            self.E_Archive_search.update(P[i], algorithm=self, problem_name=self.problem.name)
        self.pop = P

    def _mating(self, P):
        # Selection
        parents = select_parents(P)

        # Crossover
        O = self.crossover.do(self.problem, parents, P, algorithm=self)

        # Mutation
        O = self.mutation.do(self.problem, P, O, algorithm=self)

        for i in range(len(O)):
            o_F = self.evaluate(O[i].X)
            O[i].set('F', o_F)
            self.E_Archive_search.update(O[i], algorithm=self, problem_name=self.problem.name)

        return O

    def _next(self, pop):
        """
         Workflow in 'Next' step:
        + Create the offspring.
        + Select the new population.
        """
        offsprings = self._mating(pop)
        pool = pop.merge(offsprings)
        pop = self.survival.do(pool, self.pop_size)
        self.pop = pop

    def _do_each_gen(self):
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
        for arch in EA_search['X']:
            X = arch
            test_error = self.problem.get_test_performance(arch=X)
            efficiency_metric = self.problem.get_efficiency_metric(arch=X)
            F = test_error + efficiency_metric
            dummy_idv.set('X', X)
            dummy_idv.set('F', F)
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
                                       fig_name='/#Evals-IGD_search.jpg')
        visualize_IGD_value_and_nEvals(IGD_history=self.IGD_search_history,
                                       nEvals_history=self.nEvals_history,
                                       path_results=self.path_results,
                                       ylabel='IGD+ value',
                                       fig_name='/#Evals-IGDp_search.jpg')
        visualize_HV_value_and_nEvals(HV_history=self.HV_search_history,
                                      nEvals_history=self.nEvals_history,
                                      path_results=self.path_results,
                                      fig_name='/#Evals-HV_search.jpg')

if __name__ == '__main__':
    pass
