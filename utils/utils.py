import numpy as np
import random
import torch

from .compare import find_the_better
from api_benchmarks.api_101 import wrap_api as api
from api_benchmarks.api_201.utils import get_hashKey as get_hashKey_201
from api_benchmarks.api_ASR.utils import get_hashKey as get_hashKey_ASR
from api_benchmarks.api_macroNAS.utils import get_hashKey as get_hashKey_macroNAS

wrap_api = api.NASBench_(None)

def check_valid(hash_key, **kwargs):
    """
    - Check if the current solution already exists on the set of checklists.
    """
    return np.all([hash_key not in kwargs[L] for L in kwargs])

def convert_arch_genotype_int_to_api_input(genotype, problem_name):
    """
    - This function in used to convert the vector which used to representation the architecture to their representation in
    benchmarks.
    """
    if problem_name == 'NASBench201':
        OPS_LIST = ['none', 'skip_connect', 'nor_conv_1x1', 'nor_conv_3x3', 'avg_pool_3x3']
        api_input = f'|{OPS_LIST[genotype[0]]}~0|+' \
                    f'|{OPS_LIST[genotype[1]]}~0|{OPS_LIST[genotype[2]]}~1|+' \
                    f'|{OPS_LIST[genotype[3]]}~0|{OPS_LIST[genotype[4]]}~1|{OPS_LIST[genotype[5]]}~2|'
    elif problem_name == 'NASBench101':
        IDX_OPS = [1, 3, 6, 10, 15]
        edges_matrix = np.zeros((7, 7), dtype=np.int8)
        for row in range(6):
            idx_list = None
            if row == 0:
                idx_list = [2, 4, 7, 11, 16, 22]
            elif row == 1:
                idx_list = [5, 8, 12, 17, 23]
            elif row == 2:
                idx_list = [9, 13, 18, 24]
            elif row == 3:
                idx_list = [14, 19, 25]
            elif row == 4:
                idx_list = [20, 26]
            elif row == 5:
                idx_list = [27]
            for i, edge in enumerate(idx_list):
                if genotype[edge] - 1 == 0:
                    edges_matrix[row][row + i + 1] = 1

        ops_matrix = ['input']
        for i in IDX_OPS:
            if genotype[i] == 2:
                ops_matrix.append('conv1x1-bn-relu')
            elif genotype[i] == 3:
                ops_matrix.append('conv3x3-bn-relu')
            else:
                ops_matrix.append('maxpool3x3')
        ops_matrix.append('output')
        api_input = {'edges_matrix': edges_matrix, 'ops_matrix': ops_matrix}
    else:
        api_input = ''.join(map(str, genotype))
    return api_input

def get_hashKey(arch, problem_name):
    """
    This function is used to get the hash key of architecture. The hash key is used to avoid the existence of duplication in the population.\n
    - *Output*: The hash key of architecture.
    """
    arch_dummy = arch.copy()
    if problem_name in ['NASBench101', 'NASBench101*']:
        api_input = convert_arch_genotype_int_to_api_input(arch_dummy, 'NASBench101')
        edges_matrix, ops_matrix = api_input['edges_matrix'], api_input['ops_matrix']
        model_spec = api.ModelSpec(edges_matrix, ops_matrix)
        hashKey = wrap_api.get_module_hash(model_spec)
    elif problem_name in ['NASBench201', 'MacroNAS', 'NASBench301']:
        hashKey = ''.join(map(str, arch_dummy))
    elif problem_name == 'NASBench201*':
        hashKey = get_hashKey_201(arch_dummy)
    elif problem_name == 'MacroNAS*':
        hashKey = get_hashKey_macroNAS(arch_dummy)
    elif problem_name in ['NASBenchASR', 'NASBenchASR*']:
        hashKey = get_hashKey_ASR(arch_dummy)
    else:
        raise ValueError(f'Not supporting this problem - {problem_name}.')
    return hashKey

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed_all(seed)

def get_front_0(F):
    l = len(F)
    r = np.zeros(l, dtype=np.int8)
    for i in range(l):
        if r[i] == 0:
            for j in range(i + 1, l):
                better_sol = find_the_better(F[i], F[j])
                if better_sol == 0:
                    r[j] = 1
                elif better_sol == 1:
                    r[i] = 1
                    break
    return r == 0

def remove_phenotype_duplicate(genotype_list, problem_name):
    phenotype_list = []
    genotype_list_new = []
    for genotype in genotype_list:
        phenotype = get_hashKey(genotype, problem_name)
        if phenotype not in phenotype_list:
            phenotype_list.append(phenotype)
            genotype_list_new.append(genotype)
    return genotype_list_new

def run_evaluation_phase(genotype_list, problem):
    # Note: genotype_list -> have been removed duplicated ones
    fitness_list = []
    best_arch, best_performance = None, 100

    for genotype in genotype_list:
        test_performance = problem.get_test_performance(arch=genotype)
        efficiency_metric = problem.get_efficiency_metric(arch=genotype)
        F = test_performance + efficiency_metric
        fitness_list.append(F)

        if test_performance[-1] < best_performance:  # Minimization
            best_arch = convert_arch_genotype_int_to_api_input(genotype, problem.name)
            best_performance = test_performance[-1]
    fitness_list = np.array(fitness_list)

    idx_front_0 = get_front_0(fitness_list)
    AS = np.array(genotype_list)[idx_front_0]
    AS_new = [convert_arch_genotype_int_to_api_input(genotype, problem.name) for genotype in AS]
    AF = np.unique(fitness_list[idx_front_0], axis=0)

    IGD = problem.calculate_IGD(AF)
    HV = problem.calculate_HV(AF)

    if problem.name in ['NASBench101']:
        best_performance = np.round((1 - np.min(AF[:, 0])) * 100, 2)
    elif problem.name in ['NASBenchASR']:
        best_performance = np.round(np.min(AF[:, 0]) * 100, 2)
    else:
        best_performance = np.round(100 - np.min(AF[:, 0]) * 100, 2)
    opt_results = {
        'Approximation Set': AS_new,
        'Approximation Front': AF,
        'IGD': IGD,
        'HV': HV,
        'Best Architecture': best_arch,
        'Best Architecture (performance)': best_performance
    }
    return opt_results