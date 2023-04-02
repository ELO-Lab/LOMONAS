class Crossover:
    def __init__(self, n_parents, prob):
        self.n_parents = n_parents
        self.prob = prob

    def do(self, problem, parents, pop, **kwargs):
        offsprings = self._do(problem, parents, pop, **kwargs)
        return offsprings

    def _do(self, problem, parents, pop, **kwargs):
        pass
