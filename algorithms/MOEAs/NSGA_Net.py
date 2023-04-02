"""
Modified from: https://github.com/msu-coinlab/pymoo
"""
import time
import numpy as np

from algorithms.MOEAs import Algorithm


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

    def _reset(self):
        pass

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
        self.nEvals += 1
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
        while self.nEvals < self.problem.maxEvals:
            self.nGens += 1
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
            print(f'------ Gen {self.nGens} ------')
            print(f'-> nEvals / maxEvals: {self.nEvals}/{self.problem.maxEvals}')
            print()

    def log_elitist_archive(self, **kwargs):
        pass

    def _finalize(self):
        pass

if __name__ == '__main__':
    pass
