import pickle as p
import numpy as np
import argparse
from utils import get_front_0
import os
from sys import platform
from factory import get_problems
import logging
import sys


def run_transfer_evaluation(path_pre_res, path_res, problem):
    list_dir = os.listdir(path_pre_res)
    assert len(list_dir) != 1, 'Wrong path results!'
    list_IGD, list_IGDp, list_HV, list_best_arch = [], [], [], []
    for fol in list_dir:
        if fol != 'transfer_res':
            sub_path_res = path_pre_res + f'/{fol}'
            search_res = p.load(open(sub_path_res + '/search_results.p', 'rb'))
            approx_set = search_res['Approximation Set']
            F = []
            for arch in approx_set:
                test_per = problem.get_test_performance(arch)
                efficiency_metric = problem.get_efficiency_metric(arch=arch)
                f = test_per + efficiency_metric
                F.append(f)
            F = np.array(F)
            idx_fro_0 = get_front_0(F)
            approx_set_transfer = np.array(approx_set)[idx_fro_0].copy()
            approx_front_transfer = F[idx_fro_0].copy()
            approx_front_transfer = np.unique(approx_front_transfer, axis=0)
            best_arch = min(approx_front_transfer[:, 0])

            IGD_val, IGDp_val, HV_val = problem.calculate_IGD(approx_front_transfer), problem.calculate_IGDp(approx_front_transfer), problem.calculate_HV(approx_front_transfer)
            list_IGD.append(IGD_val)
            list_IGDp.append(IGDp_val)
            list_HV.append(HV_val)
            list_best_arch.append(best_arch)

            res = {
                'Approximation Set': approx_set_transfer,
                'Approximation Front': approx_front_transfer,
                'Best Performance (Error)': best_arch,
                'IGD': IGD_val, 'IGD+': IGDp_val, 'HV': HV_val
            }
            p.dump(res, open(path_res + f'/transfer_res_{fol}.p', 'wb'))
            logging.info(f'IGD (run {int(fol) + 1}): {np.round(IGD_val, 4)}')
            logging.info(f'IGD+ (run {int(fol) + 1}): {np.round(IGDp_val, 4)}')
            logging.info(f'HV (run {int(fol) + 1}): {np.round(HV_val, 4)}')
            logging.info(f'Best Performance (Error) (run {int(fol) + 1}): {best_arch}\n')
    print('=*'*70)
    logging.info(f'IGD (average): {np.round(np.mean(list_IGD), 4)} ({np.round(np.std(list_IGD), 4)})')
    logging.info(f'IGD+ (average): {np.round(np.mean(list_IGDp), 4)} ({np.round(np.std(list_IGDp), 4)})')
    logging.info(f'HV (average): {np.round(np.mean(list_HV), 4)} ({np.round(np.std(list_HV), 4)})')
    logging.info(f'Best Testing Performance (average): {np.round(np.mean(list_best_arch), 4)} ({np.round(np.std(list_best_arch), 4)})')

def main(kwargs):
    if platform == "linux" or platform == "linux2":
        root_project = '/'.join(os.path.abspath(__file__).split('/')[:-1])
    elif platform == "win32" or platform == "win64":
        root_project = '\\'.join(os.path.abspath(__file__).split('\\')[:-1])
    else:
        raise ValueError()

    if kwargs.path_pre_results is None:
        raise ValueError('Please conduct experiments on NAS-Bench-201 (CIFAR-10) or MacroNAS (CIFAR-10) first!')
    else:
        try:
            os.makedirs(f'{kwargs.path_pre_results}/transfer_res/{kwargs.problem}')
        except FileExistsError:
            pass
        PATH_PRE_RESULTS = kwargs.path_pre_results
        PATH_RESULTS = f'{kwargs.path_pre_results}/transfer_res/{kwargs.problem}'

    ''' ============================================== Set up problem ============================================== '''
    path_pof = root_project + '/data/POF'
    if kwargs.path_api_benchmark is None:
        path_api_benchmark = root_project + '/data'
    else:
        path_api_benchmark = kwargs.path_api_benchmark

    problem = get_problems(problem_name=kwargs.problem, maxEvals=0,
                           path_api_benchmark=path_api_benchmark,
                           path_pareto_optimal_front=path_pof)
    problem.set_up()

    ''' ============================================== Transfer ============================================== '''
    run_transfer_evaluation(PATH_PRE_RESULTS, PATH_RESULTS, problem)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    ''' PROBLEM '''
    parser.add_argument('--problem', type=str, default='NAS201-C10', help='the problem name',
                        choices=['NAS201-C100', 'NAS201-IN16', 'MacroNAS-C100'])

    ''' ENVIRONMENT '''
    parser.add_argument('--path_api_benchmark', type=str, default=None, help='path for loading api benchmark')
    parser.add_argument('--path_pre_results', type=str, default=None, help='path of pre-results')
    args = parser.parse_args()

    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format=log_format, datefmt='%m/%d %I:%M:%S %p')

    main(args)

