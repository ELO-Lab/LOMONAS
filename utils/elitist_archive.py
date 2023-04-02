from .compare import find_the_better
from .utils import get_hashKey
import numpy as np

class ElitistArchive:
    """
        Note: No limited the size
    """
    def __init__(self, log_each_change=True):
        self.X, self.hashKey, self.F = [], [], []
        self.size = np.inf
        self.log_each_change = log_each_change

    def update(self, idv, **kwargs):
        X = idv.X
        hashKey = get_hashKey(X, kwargs['problem_name'])
        F = idv.F

        if hashKey not in self.hashKey:
            l = len(self.F)
            r = np.zeros(l, dtype=np.int8)
            status = True
            for i, F_ in enumerate(self.F):
                better_idv = find_the_better(F, F_)
                if better_idv == 0:
                    r[i] += 1
                elif better_idv == 1:
                    status = False
                    break
            if status:
                self.X.append(X)
                self.hashKey.append(hashKey)
                self.F.append(F)
                r = np.append(r, 0)

                self.X = np.array(self.X)[r == 0].tolist()
                self.hashKey = np.array(self.hashKey)[r == 0].tolist()
                self.F = np.array(self.F)[r == 0].tolist()

            if status and self.log_each_change:
                kwargs['algorithm'].log_elitist_archive(new_idv=idv)
