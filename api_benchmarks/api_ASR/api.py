import pickle as p
from utils import get_hashKey

class API:
    def __init__(self, path_data, dataset):
        self.dataset = 'TIMIT'
        self.api = p.load(open(path_data + f'/[{dataset}]_data.p', 'rb'))

    def get_test_per(self, arch):
        keyData = get_hashKey(arch, 'NASBenchASR')
        return self.api[keyData]['test_per']

    @staticmethod
    def get_training_time():
        return 1

    def get_val_per(self, arch, iepoch=-1):
        keyData = get_hashKey(arch, 'NASBenchASR')
        return self.api[keyData]['val_per'][iepoch - 1], self.get_training_time()

    def get_efficiency_metrics(self, arch, metric):
        keyData = get_hashKey(arch, 'NASBenchASR')
        efficiency_metrics = {
            'FLOPs': self.api[keyData]['FLOPs'],
            'params': self.api[keyData]['params'],
            'latency-jetson-nano': self.api[keyData]['latency']['jetson-nano'],
        }
        return efficiency_metrics[metric]


