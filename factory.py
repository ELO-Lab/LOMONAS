from problems import NASBench101, NASBench201, MacroNAS, NASBenchASR
from operators.crossover import PointCrossover
from operators.mutation import BitStringMutation
from operators.sampling.random_sampling import RandomSampling
from operators.selection import RankAndCrowdingSurvival

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

def get_problems(problem_name, max_eval=3000, **kwargs):
    if problem_name == 'NAS101':
        return NASBench101(dataset='CIFAR-10', max_eval=max_eval, **kwargs)
    elif 'NAS201' in problem_name:
        if 'C100' in problem_name:
            dataset = 'CIFAR-100'
        elif 'C10' in problem_name:
            dataset = 'CIFAR-10'
        elif 'IN16' in problem_name:
            dataset = 'ImageNet16-120'
        else:
            raise ValueError
        return NASBench201(dataset=dataset, max_eval=max_eval, **kwargs)
    elif 'MacroNAS' in problem_name:
        if 'C100' in problem_name:
            dataset = 'CIFAR-100'
        elif 'C10' in problem_name:
            dataset = 'CIFAR-10'
        else:
            raise ValueError
        return MacroNAS(dataset=dataset, max_eval=max_eval, **kwargs)
    elif 'NAS-ASR' in problem_name:
        return NASBenchASR(dataset='TIMIT', max_eval=max_eval, **kwargs)
    else:
        raise ValueError(f'Not supporting this problem - {problem_name}.')

def get_optimizer(optimizer_name, **kwargs):
    if 'MOEA' in optimizer_name:
        pop_size = kwargs['pop_size']
        sampling = RandomSampling()
        crossover = PointCrossover('2X')
        mutation = BitStringMutation()
        if 'NSGAII' in optimizer_name:
            from algorithms.MOEAs import NSGA_Net
            survival = RankAndCrowdingSurvival()
            optimizer = NSGA_Net(name=optimizer_name)
            optimizer.set_hyperparameters(
                pop_size=pop_size,
                sampling=sampling,
                crossover=crossover,
                mutation=mutation,
                survival=survival,
                debug=bool(kwargs['debug'])
            )
        elif 'MOEAD' in optimizer_name:
            from algorithms.MOEAs import MOEAD_Net
            optimizer = MOEAD_Net(name=optimizer_name)
            optimizer.set_hyperparameters(
                pop_size=pop_size,
                sampling=sampling,
                crossover=crossover,
                mutation=mutation,
                debug=bool(kwargs['debug'])
            )
    elif optimizer_name == 'LOMONAS':
        from algorithms.MOLS import LOMONAS
        optimizer = LOMONAS(name=optimizer_name)
        optimizer.set_hyperparameters(
            NF=kwargs['NF'],
            check_all_neighbors=bool(kwargs['check_all_neighbors']),
            neighborhood_check_on_all_sols=bool(kwargs['neighborhood_check_on_all_sols']),
            debug=bool(kwargs['debug']),
        )
    elif optimizer_name == 'RR_LS':
        from algorithms.MOLS import RR_LS
        optimizer = RR_LS(name=optimizer_name)
        optimizer.set_hyperparameters(
            loop=bool(kwargs['loop']),
            debug=bool(kwargs['debug'])
        )
    else:
        raise ValueError(f'Not supporting this algorithm - {optimizer_name}.')
    return optimizer

def get_objectives(problem_name, optimizer_name):
    return objectives4algorithm[problem_name][optimizer_name]


if __name__ == '__main__':
    pass
