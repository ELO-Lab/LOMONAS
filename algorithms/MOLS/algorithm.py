from utils import (
    set_seed,
    ElitistArchive
)


class Algorithm:
    def __init__(self, **kwargs):
        """
        List of general hyperparameters:\n
        - *name*: the algorithms name
        - *problem*: the problem which are being solved
        - *seed*: random seed
        - *nEvals*: the number of evaluate function calls (or the number of trained architectures (in NAS problems))
        - *path_results*: the folder where the results will be saved on
        - *E_Archive*: the Elitist Archive
        """
        # General hyperparameters
        self.name = kwargs['name']

        self.problem = None

        self.seed = 0
        self.nEvals = 0

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
        self.seed = 0
        self.nEvals = 0

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

    """ -------------------------------- Perform when having a new change in EA -------------------------------- """
    def log_elitist_archive(self, **kwargs):
        raise NotImplementedError

    """ --------------------------------------------------- Finalize ----------------------------------------------- """
    def finalize(self):
        self._finalize()

    """ -------------------------------------------- Abstract Methods -----------------------------------------------"""
    def _setup(self):
        pass

    def _reset(self):
        pass

    def _finalize(self):
        pass

    def _solve(self):
        raise NotImplementedError

    def _evaluate(self, arch):
        raise NotImplementedError


if __name__ == '__main__':
    pass
