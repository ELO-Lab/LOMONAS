from utils import Population
from utils import get_hashKey

class RandomSampling:
    def __init__(self, nSamples=0):
        self.nSamples = nSamples

    def do(self, problem, **kwargs):
        P = Population(self.nSamples)
        n = 0

        P_hashKey = []
        while n < self.nSamples:
            X = problem.sample_a_compact_architecture()
            if problem.isValid(X):
                hashKey = get_hashKey(X, problem.name)
                P_hashKey.append(hashKey)
                P[n].set('X', X)
                P[n].set('hashKey', hashKey)
                n += 1
        return P

if __name__ == '__main__':
    pass