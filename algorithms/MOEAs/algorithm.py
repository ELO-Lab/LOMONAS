"""
Modified from: https://github.com/msu-coinlab/pymoo
"""
from utils import (
    set_seed,
    ElitistArchive
)

class Algorithm:
    def __init__(self, **kwargs):
        """
        List of general hyperparameters:\n
        - *name*: the algorithms name
        - *pop_size*: the population size
        - *sampling*: the processor of sampling process
        - *crossover*: the processor of crossover process
        - *mutation*: the processor of mutation process
        - *survival*: the processor of survival process
        - *pop*: the population
        - *problem*: the problem which are being solved
        - *seed*: random seed
        - *nEvals*: the number of evaluate function calls (or the number of trained architectures (in NAS problems))
        - *path_results*: the folder where the results will be saved on
        - *E_Archive*: the Elitist Archive
        """
        # General hyperparameters
        self.name = kwargs['name']

        self.pop_size = None
        self.sampling = None
        self.crossover = None
        self.mutation = None
        self.survival = None

        self.pop = None
        self.problem = None

        self.seed = 0
        self.nEvals, self.nGens = 0, 0

        self.path_results = None
        self.debug = False

        self.E_Archive_search = ElitistArchive(log_each_change=True)
        ##############################################################################
        self.start_executed_time_algorithm = 0.0
        self.finish_executed_time_algorithm = 0.0

        self.executed_time_algorithm_history = [0.0]
        self.benchmark_time_algorithm_history = [0.0]
        self.indicator_time_history = [0.0]
        self.evaluated_time_history = [0.0]
        self.running_time_history = [0.0]

    """ ---------------------------------- Setting Up ---------------------------------- """
    def set_hyperparameters(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def reset(self):
        self.pop = None

        self.seed = 0
        self.nEvals, self.nGens = 0, 0

        self.path_results = None

        self.E_Archive_search = ElitistArchive(log_each_change=True)

        self.start_executed_time_algorithm = 0.0
        self.finish_executed_time_algorithm = 0.0

        self.executed_time_algorithm_history = [0.0]
        self.benchmark_time_algorithm_history = [0.0]
        self.indicator_time_history = [0.0]
        self.evaluated_time_history = [0.0]
        self.running_time_history = [0.0]

        self._reset()

    """ ---------------------------------- Evaluate ---------------------------------- """
    def evaluate(self, arch):
        return self._evaluate(arch)

    """ ---------------------------------- Initialize ---------------------------------- """
    def initialize(self):
        self._initialize()

    """ ----------------------------------- Next ----------------------------------- """
    def next(self, pop):
        self._next(pop)

    """ ---------------------------------- Setting Up ---------------------------------- """
    def set_up(self, problem, seed):
        self.problem = problem
        self.seed = seed
        set_seed(self.seed)
        self._setup()

    """ ---------------------------------- Solving ---------------------------------- """
    def solve(self, problem, seed):
        self.set_up(problem, seed)
        results = self._solve()
        return results

    """ -------------------------------- Do Each Gen -------------------------------- """
    def do_each_gen(self):
        self._do_each_gen()

    """ -------------------------------- Perform when having a new change in EA -------------------------------- """
    def log_elitist_archive(self, **kwargs):
        raise NotImplementedError

    """ --------------------------------------------------- Finalize ----------------------------------------------- """
    def finalize(self):
        self._finalize()

    """ -------------------------------------------- Abstract Methods -----------------------------------------------"""
    def _solve(self):
        pass
    
    def _setup(self):
        pass

    def _reset(self):
        pass

    def _evaluate(self, arch):
        raise NotImplementedError

    def _initialize(self):
        raise NotImplementedError

    def _mating(self, P):
        raise NotImplementedError

    def _next(self, pop):
        raise NotImplementedError

    def _do_each_gen(self):
        pass

    def _finalize(self):
        pass

if __name__ == '__main__':
    pass
