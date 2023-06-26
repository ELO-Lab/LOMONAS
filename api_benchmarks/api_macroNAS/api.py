import json

OPS_LIST = ['I', '1', '2']

def arch_2_key(arch):
    key = ''
    for int_ele in arch:
        key += f'{OPS_LIST[int_ele]}'
    return key

class API:
    def __init__(self, path_data, dataset):
        self.dataset = dataset
        self.api = json.load(open(path_data + f'/[{dataset}]_data.json'))

    def get_test_accuracy(self, arch):
        keyData = arch_2_key(arch)
        return self.api[keyData]['test_acc']

    @staticmethod
    def get_training_time():
        return 1

    def get_val_accuracy(self, arch):
        keyData = arch_2_key(arch)
        return self.api[keyData]['val_acc'], self.get_training_time()

    def get_efficiency_metrics(self, arch, metric):
        keyData = arch_2_key(arch)
        efficiency_metrics = {
            'MMACs': self.api[keyData]['MMACs'],
            'params': self.api[keyData]['Params'],
        }
        return efficiency_metrics[metric]
