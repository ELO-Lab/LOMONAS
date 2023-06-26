import copy


class Individual:
    def __init__(self, x=None, hashKey=None, f=None, **kwargs) -> None:
        self.X = x
        self.F = f
        self.hashKey = hashKey
        self.data = kwargs

    def set(self, key, value):
        if key in self.__dict__:
            self.__dict__[key] = value
        else:
            self.data[key] = value

    def copy(self):
        ind = copy.copy(self)
        ind.data = self.data.copy()
        return ind

    def get(self, key):
        if key in self.__dict__:
            return self.__dict__[key]
        if key in self.data:
            return self.data[key]
        return None


if __name__ == '__main__':
    pass
