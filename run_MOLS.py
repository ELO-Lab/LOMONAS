import os
import time
import argparse
import sys
import pickle as p
import numpy as np
from sys import platform
from datetime import datetime
import logging

from factory import get_problems, get_optimizer, get_objectives
from utils import run_evaluation_phase, visualize_archive

def main(kwargs):
    if platform == "linux" or platform == "linux2":
        root_project = '/'.join(os.path.abspath(__file__).split('/')[:-1])
    elif platform == "win32" or platform == "win64":
        root_project = '\\'.join(os.path.abspath(__file__).split('\\')[:-1])
    else:
        raise ValueError()

    if kwargs.path_results is None:
        try:
            os.makedirs(f'{root_project}/exp_rs/{kwargs.problem}')
        except FileExistsError:
            pass
        PATH_RESULTS = f'{root_project}/exp_rs/{kwargs.problem}'
    else:
        try:
            os.makedirs(f'{kwargs.path_results}/{kwargs.problem}')
        except FileExistsError:
            pass
        PATH_RESULTS = f'{kwargs.path_results}/{kwargs.problem}'

    ''' ============================================== Set up problem ============================================== '''
    path_pof = root_project + '/data/POF'
    if kwargs.path_api_benchmark is None:
        path_api_benchmark = root_project + '/data'
    else:
        path_api_benchmark = kwargs.path_api_benchmark

    problem = get_problems(problem_name=kwargs.problem, maxEvals=kwargs.maxEvals,
                           path_api_benchmark=path_api_benchmark,
                           path_pareto_optimal_front=path_pof)
    problem.set_up()

    ''' ==================================================================================================== '''
    n_runs = kwargs.n_runs
    init_seed = kwargs.seed

    optimizer = get_optimizer(optimizer_name=kwargs.optimizer)

    objectives = get_objectives(problem.name, optimizer.name)
    optimizer.set_hyperparameters(debug=bool(kwargs.debug),
                                  f0=objectives['f0'],
                                  f1=objectives['f1'])
    if 'LOMONAS' in optimizer.name:
        NF = kwargs.NF
        get_all_neighbors = bool(kwargs.get_all_neighbors)
        local_search_on_all_sols = bool(kwargs.local_search_on_all_sols)
        optimizer.set_hyperparameters(
            NF=NF,
            get_all_neighbors=get_all_neighbors,
            local_search_on_all_sols=local_search_on_all_sols
        )
    else:  # RR-LS
        loop = bool(kwargs.loop)
        optimizer.set_hyperparameters(
            loop=loop
        )

    ''' ==================================== Set up experimental environment ======================================= '''
    time_now = datetime.now()

    if 'LOMONAS' in optimizer.name:
        dir_name = time_now.strftime(
            f'{kwargs.problem}_'
            f'{optimizer.name}_NF={optimizer.NF}_{optimizer.get_all_neighbors}_{optimizer.local_search_on_all_sols}_'
            f'd%d_m%m_H%H_M%M_S%S')
    else:
        dir_name = time_now.strftime(
            f'{kwargs.problem}_'
            f'{optimizer.name}_'
            f'd%d_m%m_H%H_M%M_S%S')

    root_path = PATH_RESULTS + '/' + dir_name
    os.mkdir(root_path)
    logging.info(f'--> Create folder {root_path} - Done\n')
    logging.info(f'--> Experimental results are logged in {root_path}.')

    random_seeds_list = [init_seed + run * 100 for run in range(n_runs)]
    executed_time_list = []

    ''' =============================================== Log Information ============================================ '''
    logging.info(f'******* PROBLEM *******')
    logging.info(f'- Benchmark: {problem.name}')
    logging.info(f'- Dataset: {problem.dataset}')
    logging.info(f'- Maximum number of evaluations: {problem.maxEvals}')
    logging.info(f'- The first objective (minimize): {problem.objective_0}')
    logging.info(f'- The second objective (minimize): {problem.objective_1}\n')

    logging.info(f'******* OPTIMIZER *******')
    logging.info(f'- Algorithm name: {optimizer.name}\n')

    if 'LOMONAS' in optimizer.name:
        logging.info(f'- NF: {optimizer.NF}')
        logging.info(f'- Evaluate all neighbors?: {optimizer.get_all_neighbors}')
        logging.info(f'- Local search on all solutions?: {optimizer.local_search_on_all_sols}\n')
    else:
        logging.info(f'- Loop: {optimizer.loop}\n')

    logging.info(f'- The first objective (minimize): {optimizer.f0}')
    logging.info(f'- The second objective (minimize): {optimizer.f1}\n')

    logging.info(f'******* ENVIRONMENT *******')

    logging.info(f'- Number of running experiments: {n_runs}')
    logging.info(f'- Random seed each run: {random_seeds_list}')
    logging.info(f'- Path for saving results: {root_path}')
    logging.info(f'- Debug: {optimizer.debug}\n')

    ''' ==================================================================================================== '''
    list_IGD = []
    list_HV = []
    list_best_acc = []
    list_search_cost = []

    list_IGD_s, list_HV_s = [], []
    for run_i in range(n_runs):
        logging.info(f'-------------------------------------- Run {run_i + 1}/{n_runs} --------------------------------------')
        logging.info(f'-------------------------------------- SEARCH PHASE --------------------------------------\n')
        optimizer.reset()

        random_seed = random_seeds_list[run_i]

        path_results = root_path + '/' + f'{run_i}'

        os.mkdir(path_results)
        s = time.time()

        optimizer.set_hyperparameters(path_results=path_results)

        # SEARCH PHASE
        search_results = optimizer.solve(problem, random_seed)
        p.dump(search_results, open(f'{path_results}/search_results.p', 'wb'))

        executed_time = time.time() - s
        executed_time_list.append(executed_time)
        optimizer.problem.reset()
        AF_s = search_results['Approximation Front']
        visualize_archive(AF_s, xlabel=problem.objective_1, ylabel=problem.objective_0 + '(val_per)',
                          label=f'{optimizer.name}', path=path_results, fig_name='approximation_front_search')
        IGD_s, HV_s = optimizer.problem.calculate_IGD_val(AF_s), optimizer.problem.calculate_HV(AF_s)
        list_IGD_s.append(IGD_s)
        list_HV_s.append(HV_s)
        logging.info(f'IGD (run {run_i + 1}): {IGD_s}')
        logging.info(f'HV (run {run_i + 1}): {HV_s}')
        logging.info(f'This run takes {executed_time_list[-1]} seconds\n')

        # EVALUATION PHASE
        logging.info(f'-------------------------------------- EVALUATION PHASE --------------------------------------\n')
        genotype_list = search_results['Approximation Set']

        evaluation_results = run_evaluation_phase(genotype_list, problem)
        p.dump(evaluation_results, open(f'{path_results}/evaluation_results.p', 'wb'))
        AF = evaluation_results['Approximation Front']
        visualize_archive(AF, xlabel=problem.objective_1, ylabel=problem.objective_0,
                          label=f'{optimizer.name}', path=path_results, fig_name='approximation_front')
        logging.info(f'Approximation Set:')
        for arch in evaluation_results["Approximation Set"]:
            logging.info(f'{arch}')
        print('*' * 50)
        logging.info(f'IGD (run {run_i + 1}): {evaluation_results["IGD"]}')
        logging.info(f'HV (run {run_i + 1}): {evaluation_results["HV"]}')
        logging.info(
            f'Best Testing Performance (Accuracy|PER): {evaluation_results["Best Architecture (performance)"]}')
        logging.info(f'Search Cost: {search_results["Search Cost"]}\n')
        list_IGD.append(evaluation_results['IGD'])
        list_HV.append(evaluation_results['HV'])
        list_best_acc.append(evaluation_results['Best Architecture (performance)'])
        list_search_cost.append(search_results['Search Cost'])
    print('=*'*70)
    logging.info(f'IGD search (average): {np.round(np.mean(list_IGD_s), 4)} ({np.round(np.std(list_IGD_s), 4)})')
    logging.info(f'HV search (average): {np.round(np.mean(list_HV_s), 4)} ({np.round(np.std(list_HV_s), 4)})')
    logging.info(f'IGD (average): {np.round(np.mean(list_IGD), 4)} ({np.round(np.std(list_IGD), 4)})')
    logging.info(f'HV (average): {np.round(np.mean(list_HV), 4)} ({np.round(np.std(list_HV), 4)})')
    logging.info(f'Best Testing Performance (average): {np.round(np.mean(list_best_acc), 2)} ({np.round(np.std(list_best_acc), 2)})')
    logging.info(f'Search Cost (average): {np.round(np.mean(list_search_cost))}\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    ''' PROBLEM '''
    parser.add_argument('--problem', type=str, default='NAS201-C10', help='the problem name',
                        choices=['NAS101', 'NAS201-C10', 'NAS201-C100', 'NAS201-IN16',
                                 'MacroNAS-C10', 'MacroNAS-C100', 'NAS-ASR'])
    parser.add_argument('--maxEvals', type=int, default=3000, help='the maximum number of evaluations')

    ''' ALGORITHM '''
    parser.add_argument('--optimizer', type=str, default='LOMONAS', help='the optimizer',
                        choices=['LOMONAS', 'RR_LS'])
    parser.add_argument('--NF', type=int, default=3, help='the number of selected front for local search')
    parser.add_argument('--get_all_neighbors', type=int, default=0, help='check all neighbors?')
    parser.add_argument('--local_search_on_all_sols', type=int, default=0, help='perform local search on all solutions?')
    parser.add_argument('--loop', type=int, default=0, help='RR-LS with loop')

    ''' ENVIRONMENT '''
    parser.add_argument('--path_api_benchmark', type=str, default=None, help='path for loading api benchmark')
    parser.add_argument('--path_results', type=str, default=None, help='path for saving results')
    parser.add_argument('--n_runs', type=int, default=31, help='number of experiment runs')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--debug', type=int, default=0, help='debug mode')
    args = parser.parse_args()

    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format=log_format, datefmt='%m/%d %I:%M:%S %p')

    main(args)
