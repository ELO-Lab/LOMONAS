from problems import NASBench101, NASBench201, MacroNAS, NASBenchASR

objectives4algorithm = {
    'NASBench101': {
        'MOEA_NSGAII': {'f0': 'val_error_12', 'f1': '#params'},
        'MOEA_MOEAD': {'f0': 'val_error_12', 'f1': '#params'},
        'LOMONAS': {'f0': 'val_error_12', 'f1': '#params'},
        'RR_LS': {'f0': 'val_error_12', 'f1': '#params'},
    },
    'NASBench201': {
        'MOEA_NSGAII': {'f0': 'val_error_12', 'f1': 'FLOPs'},
        'MOEA_MOEAD': {'f0': 'val_error_12', 'f1': 'FLOPs'},
        'LOMONAS': {'f0': 'val_error_12', 'f1': 'FLOPs'},
        'RR_LS': {'f0': 'val_error_12', 'f1': 'FLOPs'},
    },
    'MacroNAS': {
        'MOEA_NSGAII': {'f0': 'val_error_-1', 'f1': 'MMACs'},
        'MOEA_MOEAD': {'f0': 'val_error_-1', 'f1': 'MMACs'},
        'LOMONAS': {'f0': 'val_error_-1', 'f1': 'MMACs'},
        'RR_LS': {'f0': 'val_error_-1', 'f1': 'MMACs'},
    },
    'NASBenchASR': {
        'MOEA_NSGAII': {'f0': 'val_per_12', 'f1': 'FLOPs'},
        'MOEA_MOEAD': {'f0': 'val_per_12', 'f1': 'FLOPs'},
        'LOMONAS': {'f0': 'val_per_12', 'f1': 'FLOPs'},
        'RR_LS': {'f0': 'val_per_12', 'f1': 'FLOPs'},
    },
}

def get_problems(problem_name, maxEvals=3000, **kwargs):
    if problem_name == 'NAS101':
        return NASBench101(dataset='CIFAR-10', maxEvals=maxEvals, **kwargs)
    elif 'NAS201' in problem_name:
        if 'C100' in problem_name:
            dataset = 'CIFAR-100'
        elif 'C10' in problem_name:
            dataset = 'CIFAR-10'
        elif 'IN16' in problem_name:
            dataset = 'ImageNet16-120'
        else:
            raise ValueError
        return NASBench201(dataset=dataset, maxEvals=maxEvals, **kwargs)
    elif 'MacroNAS' in problem_name:
        if 'C100' in problem_name:
            dataset = 'CIFAR-100'
        elif 'C10' in problem_name:
            dataset = 'CIFAR-10'
        else:
            raise ValueError
        return MacroNAS(dataset=dataset, maxEvals=maxEvals, **kwargs)
    elif 'NAS-ASR' in problem_name:
        return NASBenchASR(dataset='TIMIT', maxEvals=maxEvals, **kwargs)
    else:
        raise ValueError(f'Not supporting this problem - {problem_name}.')

def get_optimizer(optimizer_name, **kwargs):
    if optimizer_name == 'MOEA_NSGAII':
        from algorithms.MOEAs import NSGA_Net
        return NSGA_Net(name=optimizer_name)
    elif optimizer_name == 'MOEA_MOEAD':
        from algorithms.MOEAs import MOEAD_Net
        return MOEAD_Net(name=optimizer_name)
    elif optimizer_name == 'LOMONAS':
        from algorithms.MOLS import LOMONAS
        return LOMONAS(name=optimizer_name)
    elif optimizer_name == 'RR_LS':
        from algorithms.MOLS import RR_LS
        return RR_LS(name=optimizer_name)
    else:
        raise ValueError(f'Not supporting this algorithm - {optimizer_name}.')

def get_objectives(problem_name, optimizer_name):
    return objectives4algorithm[problem_name][optimizer_name]


if __name__ == '__main__':
    pass
