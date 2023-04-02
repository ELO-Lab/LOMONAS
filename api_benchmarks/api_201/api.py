import pickle as p

def arch_2_key(arch):
    return ''.join(map(str, arch))

class API:
    def __init__(self, data_path, dataset):
        self.dataset = dataset
        self.api = p.load(open(data_path + f'/[{dataset}]_data.p', 'rb'))

    def get_test_accuracy(self, arch, iepoch=-1):
        keyData = arch_2_key(arch)
        return self.api['200'][keyData]['test_acc'][iepoch]

    def get_training_time(self, keyData, iepoch):
        if self.dataset == 'CIFAR-10':
            return self.api['200'][keyData]['train_time'] / 2 * iepoch
        return self.api['200'][keyData]['train_time'] * iepoch

    def get_training_loss(self, arch, iepoch=-1):
        keyData = arch_2_key(arch)
        return self.api['200'][keyData]['train_loss'][iepoch - 1], self.get_training_time(keyData=keyData, iepoch=iepoch)

    def get_val_accuracy(self, arch, iepoch=-1):
        keyData = arch_2_key(arch)
        return self.api['200'][keyData]['val_acc'][iepoch - 1], self.get_training_time(keyData=keyData, iepoch=iepoch)

    def get_efficiency_metrics(self, arch, metric):
        keyData = arch_2_key(arch)
        efficiency_metrics = {
            'FLOPs': self.api['200'][keyData]['FLOPs'],
            'params': self.api['200'][keyData]['params'],
            'latency': self.api['200'][keyData]['latency'],
        }
        return efficiency_metrics[metric]
