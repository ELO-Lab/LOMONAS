from utils import (
    set_seed,
    ElitistArchive,
    visualize_Elitist_Archive_and_Pareto_Front,
    visualize_IGD_value_and_nEvals,
    visualize_HV_value_and_nEvals
)

import numpy as np
import pickle as p

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
        self.E_Archive_search_each_gen = []
        self.E_Archive_search_history = []

        ##############################################################################
        self.start_executed_time_algorithm = 0.0
        self.finish_executed_time_algorithm = 0.0

        self.executed_time_algorithm_history = [0.0]
        self.benchmark_time_algorithm_history = [0.0]
        self.indicator_time_history = [0.0]
        self.evaluated_time_history = [0.0]
        self.running_time_history = [0.0]

        self.nEvals_history = []

        self.E_Archive_evaluate_history = []
        self.IGD_evaluate_history = []
        self.HV_evaluate_history = []

        self.E_Archive_evaluate_each_gen = []

    """ ---------------------------------- Setting Up ---------------------------------- """
    def set_hyperparameters(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def reset(self):
        self.seed = 0
        self.nEvals = 0

        self.path_results = None

        self.E_Archive_search = ElitistArchive(log_each_change=True)
        self.E_Archive_search_history = []

        self.start_executed_time_algorithm = 0.0
        self.finish_executed_time_algorithm = 0.0

        self.executed_time_algorithm_history = [0.0]
        self.benchmark_time_algorithm_history = [0.0]
        self.indicator_time_history = [0.0]
        self.evaluated_time_history = [0.0]
        self.running_time_history = [0.0]

        self.nEvals_history = []

        self.E_Archive_evaluate_history = []

        self.IGD_evaluate_history = []
        self.HV_evaluate_history = []

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
        p.dump([self.nEvals_history, self.IGD_evaluate_history],
               open(f'{self.path_results}/#Evals_and_IGD.p', 'wb'))
        p.dump([self.nEvals_history, self.HV_evaluate_history],
               open(f'{self.path_results}/#Evals_and_HV.p', 'wb'))

        self.running_time_history = np.array(self.running_time_history)[1:]
        p.dump(self.running_time_history, open(f'{self.path_results}/running_time.p', 'wb'))

        p.dump([self.nEvals_history, self.E_Archive_search_history],
               open(f'{self.path_results}/#Evals_and_Elitist_Archive_search.p', 'wb'))
        p.dump([self.nEvals_history, self.E_Archive_evaluate_history],
               open(f'{self.path_results}/#Evals_and_Elitist_Archive_evaluate.p', 'wb'))

        visualize_Elitist_Archive_and_Pareto_Front(AF=self.E_Archive_evaluate_history[-1]['F'],
                                                   POF=self.problem.opt_pareto_front,
                                                   ylabel='Test Performance',
                                                   xlabel=self.problem.objective_1,
                                                   path=self.path_results)

        visualize_IGD_value_and_nEvals(IGD_history=self.IGD_evaluate_history,
                                       nEvals_history=self.nEvals_history,
                                       path_results=self.path_results)
        visualize_HV_value_and_nEvals(HV_history=self.HV_evaluate_history,
                                      nEvals_history=self.nEvals_history,
                                      path_results=self.path_results)
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
