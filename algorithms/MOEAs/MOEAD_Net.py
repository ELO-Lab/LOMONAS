"""
Modified from: https://github.com/msu-coinlab/pymoo
"""
import time
import numpy as np

from algorithms.MOEAs import NSGA_Net
from pymoo.util.reference_direction import UniformReferenceDirectionFactory
from scipy.spatial.distance import cdist
from pymoo.decomposition.tchebicheff import Tchebicheff

def select_parents(pop, n_parents, prob_neighbor_mating, neighbors):
    pop_size = len(pop)
    if np.random.random() < prob_neighbor_mating:
        idx = np.random.choice(neighbors, n_parents, replace=False)
    else:
        idx = np.random.permutation(pop_size)[:n_parents]

    parents_pool = []
    for i in range(n_parents):
        parents_pool.append(pop[idx[i]])

    return np.array(parents_pool)

class MOEAD_Net(NSGA_Net):
    """
    MOEAD-Net (validation error rate - efficiency metrics)
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.prob_neighbor_mating = 0.9

        self.decomposition = None
        self.ideal = None
        self.ref_dirs = None
        self.neighbors = None

    def _reset(self):
        self.decomposition = None
        self.ideal = None

    def _setup(self):
        self.sampling.nSamples = self.pop_size
        attr_perf_metric = self.f0.split('_')
        self.epoch = int(attr_perf_metric[-1])

        self.n_neighbors = int(self.pop_size//4)

    def _solve(self):
        self.start_executed_time_algorithm = time.time()
        self.initialize()

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

        n_obj = 2
        self.ref_dirs = UniformReferenceDirectionFactory(n_obj, n_partitions=self.pop_size-1).do()

        # neighbours includes the entry by itself intentionally for the survival method
        self.neighbors = np.argsort(cdist(self.ref_dirs, self.ref_dirs), axis=1, kind='quicksort')[:,
                         :self.n_neighbors]

        self.decomposition = Tchebicheff()
        self.ideal = np.min(self.pop.get('F'), axis=0)

    def _mating(self, P):
        # iterate for each member of the population in random order
        for k in np.random.permutation(len(P)):
            # all neighbors of this individual and corresponding weights
            neighbors = self.neighbors[k]

            parents = select_parents(pop=self.pop, n_parents=2,
                                     prob_neighbor_mating=self.prob_neighbor_mating, neighbors=neighbors)

            # crossover
            O = self.crossover.do(self.problem, parents, P, algorithm=self)

            # mutation
            O = self.mutation.do(self.problem, P, O, algorithm=self)

            o = np.random.choice(O)

            o_F = self.evaluate(o.X)
            o.set('F', o_F)
            self.E_Archive_search.update(o, algorithm=self, problem_name=self.problem.name)

            # update the ideal point
            self.ideal = np.min(np.vstack([self.ideal, o_F]), axis=0)
            # calculate the decomposed values for each neighbor
            FV = self.decomposition.do(P[neighbors].get('F'), weights=self.ref_dirs[neighbors, :], ideal_point=self.ideal)
            off_FV = self.decomposition.do(np.array(o.F)[None, :], weights=self.ref_dirs[neighbors, :], ideal_point=self.ideal)

            # get the absolute index in F where offspring is better than the current F (decomposed space)
            I = np.where(off_FV < FV)[0]
            P[neighbors[I]] = o
        return P

    def _next(self, pop):
        pop = self._mating(pop)
        self.pop = pop

if __name__ == '__main__':
    pass
