import os
import time
import argparse
import sys
import pickle as p
import numpy as np
from sys import platform
import logging
import json

from factory import get_problems, get_optimizer, get_objectives
from utils import run_evaluation_phase, visualize_archive

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

np.seterr(invalid='ignore')
def run(kwargs):
    if platform == "linux" or platform == "linux2":
        PROJECT_PATH = '/'.join(os.path.abspath(__file__).split('/')[:-1])
    elif platform == "win32" or platform == "win64":
        PROJECT_PATH = '\\'.join(os.path.abspath(__file__).split('\\')[:-1])
    else:
        raise ValueError()

    if kwargs.res_path is None:
        try:
            os.makedirs(f'{PROJECT_PATH}/exp_res/{kwargs.problem}')
        except FileExistsError:
            pass
        RES_PATH = f'{PROJECT_PATH}/exp_res/{kwargs.problem}'
    else:
        try:
            os.makedirs(f'{kwargs.res_path}/{kwargs.problem}')
        except FileExistsError:
            pass
        RES_PATH = f'{kwargs.res_path}/{kwargs.problem}'

    ''' ============================================== Set up problem ============================================== '''
    POF_PATH = PROJECT_PATH + '/data/POF'

    if kwargs.api_benchmark_path is None:
        API_BENCHMARK_PATH = PROJECT_PATH + '/data'
    else:
        API_BENCHMARK_PATH = kwargs.api_benchmark_path
    problem = get_problems(problem_name=kwargs.problem, max_eval=kwargs.max_eval,
                           api_benchmark_path=API_BENCHMARK_PATH,
                           pof_path=POF_PATH)
    problem.set_up()

    ''' ==================================================================================================== '''
    n_run = kwargs.n_run
    init_seed = kwargs.init_seed

    optimizer = get_optimizer(
        optimizer_name=kwargs.optimizer,
        NF=kwargs.NF,
        check_all_neighbors=bool(kwargs.check_all_neighbors),
        neighborhood_check_on_all_sols=bool(kwargs.neighborhood_check_on_all_sols),
        pop_size=kwargs.pop_size,
        loop=bool(kwargs.loop),
        debug=bool(kwargs.debug)
    )
    objectives = get_objectives(problem.name, optimizer.name)
    optimizer.set_hyperparameters(f0=objectives['f0'], f1=objectives['f1'])

    ''' ==================================== Set up experimental environment ======================================= '''
    dir_name = f'{kwargs.problem}_{optimizer.name}'
    ALGO_RES_PATH = RES_PATH + '/' + dir_name
    try:
        os.mkdir(ALGO_RES_PATH)
    except FileExistsError:
        pass
    logging.info(f'--> Experimental results are logged in {ALGO_RES_PATH}.')

    executed_time_list = []
    list_IGD_s, list_IGDp_s, list_HV_s = [], [], []
    list_IGD, list_IGDp, list_HV = [], [], []
    list_best_acc = []
    list_search_cost = []

    for rid in range(n_run):
        logging.info('\033[95m' + f'Run {rid + 1}/{n_run}' + '\033[00m')
        optimizer.reset()

        seed = init_seed + 100 * rid

        RID_RES_PATH = ALGO_RES_PATH + '/' + f'{rid}'
        try:
            os.mkdir(RID_RES_PATH)
        except FileExistsError:
            pass

        s = time.time()

        optimizer.set_hyperparameters(path_results=RID_RES_PATH)

        configuration = {
            'Problem': {
                'Benchmark': problem.name,
                'Dataset': problem.dataset,
                'Maximum #Eval': problem.max_eval,
                'Objective #1': problem.objective_0,
                'Objective #2': problem.objective_1
            },
            'Optimizer': {
                'Name': optimizer.name,
                'Objective #0': optimizer.f0,
                'Objective #1': optimizer.f1
            },
            'Environment': {
                'ID Run': rid,
                'Result Path': RID_RES_PATH,
                'Debug Mode': f'{optimizer.debug}'
            }
        }
        if 'MOEA' in kwargs.optimizer:
            configuration['Optimizer']['Pop Size'] = kwargs.pop_size
            configuration['Optimizer']['Crossover'] = optimizer.crossover.method
            configuration['Optimizer']['Mutation'] = 'Integer-encoding mutation'
            if 'NSGAII' in optimizer.name:
                configuration['Optimizer']['Selection'] = 'Ranking and Crowding Distance'
        else:
            if 'LOMONAS' in optimizer.name:
                configuration['Optimizer']['NF'] = optimizer.NF
                configuration['Optimizer']['Check all neighbors?'] = f'{optimizer.check_all_neighbors}'
                configuration['Optimizer']['Neighborhood check on all neighbors?'] = f'{optimizer.neighborhood_check_on_all_sols}'
            else:
                configuration['Optimizer']['Loop?'] = f'{optimizer.loop}'

        with open(f'{RID_RES_PATH}/configuration.json', 'w') as fp:
            json.dump(configuration, fp, indent=4, cls=NumpyEncoder)

        # SEARCH PHASE
        print("-" * 150)
        content = ['#Evals', 'IGD (search)', 'IGD+ (search)', 'HV (search)', 'IGD (evaluation)', 'IGD+ (evaluation)', 'HV (evaluation)']
        print(
            "\033[95m{:<10}\033[00m | \033[95m{:^20}\033[00m | \033[95m{:^20}\033[00m | \033[95m{:^20}\033[00m | \033[95m{:^20}\033[00m | \033[95m{:^20}\033[00m | \033[95m{:^20}\033[00m |".format(
                *content))
        search_results = optimizer.solve(problem, seed)
        p.dump(search_results, open(f'{RID_RES_PATH}/search_results.p', 'wb'))

        executed_time = time.time() - s
        executed_time_list.append(executed_time)
        optimizer.problem.reset()
        AF_s = search_results['Approximation Front']
        visualize_archive(AF_s, xlabel=problem.objective_1, ylabel='Validation Performance', title=problem.name,
                          label=f'{optimizer.name}', path=RID_RES_PATH, fig_name='approximation_front_search')
        IGD_s, IGDp_s, HV_s = optimizer.problem.calculate_IGD_val(AF_s), optimizer.problem.calculate_IGDp_val(AF_s), optimizer.problem.calculate_HV(AF_s)
        list_IGD_s.append(IGD_s)
        list_IGDp_s.append(IGDp_s)
        list_HV_s.append(HV_s)

        # EVALUATION PHASE
        genotype_list = search_results['Approximation Set']

        evaluation_results = run_evaluation_phase(genotype_list, problem)
        p.dump(evaluation_results, open(f'{RID_RES_PATH}/evaluation_results.p', 'wb'))
        AF = evaluation_results['Approximation Front']
        visualize_archive(AF, xlabel=problem.objective_1, ylabel=problem.objective_0, title=problem.name,
                          label=f'{optimizer.name}', path=RID_RES_PATH, fig_name='approximation_front')
        print("-" * 150)
        content = ['Final', IGD_s, IGDp_s, HV_s, evaluation_results["IGD"], evaluation_results["IGD+"], evaluation_results["HV"]]
        print(
            "\033[92m{:<10}\033[00m | \033[96m{:^20.6f}\033[00m | \033[96m{:^20.6f}\033[00m | \033[96m{:^20.6f}\033[00m | \033[93m{:^20.6f}\033[00m | \033[93m{:^20.6f}\033[00m | \033[93m{:^20.6f}\033[00m |".format(
                *content))
        print("-" * 150)

        print('\033[95m' + 'Approximation Set' + '\033[00m')
        for i, arch in enumerate(evaluation_results["Approximation Set"]):
            print(f'    Network #{i}: {arch}')
        print('\033[95m' + 'Best Network' + '\033[00m' + f': {evaluation_results["Best Architecture (performance)"]}%')
        print('\033[95m' + 'Search Cost' + '\033[00m' + f': {round(search_results["Search Cost"])} seconds', '\n')

        list_IGD.append(evaluation_results['IGD'])
        list_IGDp.append(evaluation_results['IGD+'])
        list_HV.append(evaluation_results['HV'])
        list_best_acc.append(evaluation_results['Best Architecture (performance)'])
        list_search_cost.append(search_results['Search Cost'])
        
        res = {
            'Run': rid,
            'IGD (search)': IGD_s,
            'IGD+ (search)': IGDp_s,
            'HV (search)': HV_s,
            'IGD': evaluation_results['IGD'],
            'IGD+': evaluation_results['IGD+'],
            'HV': evaluation_results['HV'],
        }
        with open(f'{RID_RES_PATH}/exp_result.json', 'w') as fp:
            json.dump(res, fp, indent=4, cls=NumpyEncoder)

    print("-" * 150)
    content = ['-', 'IGD (search)', 'IGD+ (search)', 'HV (search)', 'IGD (evaluation)', 'IGD+ (evaluation)', 'HV (evaluation)']
    print(
        "\033[95m{:<10}\033[00m | \033[95m{:^20}\033[00m | \033[95m{:^20}\033[00m | \033[95m{:^20}\033[00m | \033[95m{:^20}\033[00m | \033[95m{:^20}\033[00m | \033[95m{:^20}\033[00m |".format(
            *content))
    print("-" * 150)
    IGD_s_avg = f'{np.round(np.mean(list_IGD_s), 4)} ({np.round(np.std(list_IGD_s), 4)})'
    IGDp_s_avg = f'{np.round(np.mean(list_IGDp_s), 4)} ({np.round(np.std(list_IGDp_s), 4)})'
    HV_s_avg = f'{np.round(np.mean(list_HV_s), 4)} ({np.round(np.std(list_HV_s), 4)})'
    IGD_avg = f'{np.round(np.mean(list_IGD), 4)} ({np.round(np.std(list_IGD), 4)})'
    IGDp_avg = f'{np.round(np.mean(list_IGDp), 4)} ({np.round(np.std(list_IGDp), 4)})'
    HV_avg = f'{np.round(np.mean(list_HV), 4)} ({np.round(np.std(list_HV), 4)})'
    content = ['Average', IGD_s_avg, IGDp_s_avg, HV_s_avg, IGD_avg, IGDp_avg, HV_avg]
    print(
        "\033[92m{:<10}\033[00m | \033[96m{:^20}\033[00m | \033[96m{:^20}\033[00m | \033[96m{:^20}\033[00m | \033[93m{:^20}\033[00m | \033[93m{:^20}\033[00m | \033[93m{:^20}\033[00m |".format(
            *content))
    print("-" * 150)
    print('\033[95m' + 'Best Network' + '\033[00m' + f': {np.round(np.mean(list_best_acc), 2)} ({np.round(np.std(list_best_acc), 2)})')
    print('\033[95m' + 'Search Cost' + '\033[00m' + f': {int(np.mean(list_search_cost))} seconds')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    ''' PROBLEM '''
    parser.add_argument('--problem', type=str, default='NAS201-C10', help='the problem name',
                        choices=['NAS101', 'NAS201-C10', 'NAS201-C100', 'NAS201-IN16',
                                 'MacroNAS-C10', 'MacroNAS-C100', 'NAS-ASR'])
    parser.add_argument('--max_eval', type=int, default=3000, help='the maximum number of evaluations')

    ''' ALGORITHM '''
    parser.add_argument('--optimizer', type=str, default='LOMONAS', help='the optimizer',
                        choices=['MOEA_NSGAII', 'MOEA_MOEAD', 'LOMONAS', 'RR_LS'])
    parser.add_argument('--pop_size', type=int, default=20, help='the population size')
    parser.add_argument('--NF', type=int, default=3, help='the number of selected fronts for neighborhood check')
    parser.add_argument('--check_all_neighbors', type=int, default=0, help='check all neighbors?')
    parser.add_argument('--neighborhood_check_on_all_sols', type=int, default=0, help='perform neighborhood check on all solutions?')
    parser.add_argument('--loop', type=int, default=0, help='RR-LS with loop')

    ''' ENVIRONMENT '''
    parser.add_argument('--api_benchmark_path', type=str, default=None, help='path for loading api benchmark')
    parser.add_argument('--res_path', type=str, default=None, help='path for saving results')
    parser.add_argument('--n_run', type=int, default=31, help='number of experiment runs')
    parser.add_argument('--init_seed', type=int, default=0, help='random seed')
    parser.add_argument('--debug', type=int, default=0, help='debug mode')
    args = parser.parse_args()

    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format=log_format, datefmt='%m/%d %I:%M:%S %p')

    run(args)