import numpy as np
import pickle as p
from api_benchmarks.api_macroNAS.api import API
from problems.NAS_problem import Problem
from utils import get_hashKey
from pymoo.indicators.hv import HV
from pymoo.indicators.igd import IGD
from pymoo.indicators.igd_plus import IGDPlus

# 'Identity': 0
# 'MBConv3_3x3': 1
# 'MBConv6_5x5': 2
OPS_LIST = ['I', '1', '2']

reference_point = [1.01, 1.01]
HV_cal = HV(reference_point)

class MacroNAS(Problem):
    def __init__(self, dataset, max_eval, **kwargs):
        """
        # MacroNAS provides us the information (e.g., the testing accuracy,the validation accuracy,
        the number of MMACs, etc) of all architectures in the search space. Therefore, if we
        want to evaluate any architectures in the search space, we just need to query its information in the data.\n
        -----------------------------------------------------------------

        # Additional Hyper-parameters:\n
        - data_path -> the path contains NAS-Bench-201 data.
        - available_ops -> the available operators can choose in the search space.
        - maxLength -> the maximum length of compact architecture.
        """

        super().__init__(max_eval, 'MacroNAS', dataset, **kwargs)
        self.objective_0 = 'Test Error'
        self.objective_1 = '#MMACs'

        self.available_ops = [0, 1, 2]
        self.maxLength = 14

        self.path_data = kwargs['api_benchmark_path'] + f'/MacroNAS'
        self.pof_path = kwargs['pof_path']

        self.min_MMACs, self.max_MMACs = None, None

        self.opt_pareto_front = None
        self.tf_estimator = None

        self.test_performance_cache = {}
        self.training_based_performance_cache = {}
        self.training_free_performance_cache = {}
        self.efficiency_cache = {}

    def _set_up(self):
        available_subsets = ['CIFAR-10', 'CIFAR-100']
        if self.dataset not in available_subsets:
            raise ValueError(f'Just only supported these subsets: CIFAR-10; CIFAR-100.'
                             f'{self.dataset} subset is not supported at this time.')

        self.api = API(path_data=self.path_data, dataset=self.dataset)

        if self.dataset == 'CIFAR-10':
            self.min_MMACs, self.max_MMACs = 21.31, 239.28
        else:
            self.min_MMACs, self.max_MMACs = 21.54, 239.51

        f_opt_pareto_front = open(f'{self.pof_path}/[POF_TestAcc_MMACs]_[MacroNAS_{self.dataset}].p', 'rb')
        self.opt_pareto_front = p.load(f_opt_pareto_front)
        self.opt_pareto_front[:, 0] /= 100
        f_opt_pareto_front.close()

        self.opt_pareto_front_norm = self.opt_pareto_front.copy()
        self.opt_pareto_front_norm[:, 1] = np.round(
            (self.opt_pareto_front_norm[:, 1] - self.min_MMACs) / (self.max_MMACs - self.min_MMACs), 4)
        self.opt_pareto_front_norm = np.round(self.opt_pareto_front_norm, 6)

        f_opt_pareto_front_val = open(f'{self.pof_path}/[POF_ValAcc_MMACs]_[MacroNAS_{self.dataset}].p', 'rb')
        self.opt_pareto_front_val = p.load(f_opt_pareto_front_val)
        self.opt_pareto_front_val[:, 0] /= 100
        f_opt_pareto_front_val.close()

        self.opt_pareto_front_val_norm = self.opt_pareto_front_val.copy()
        self.opt_pareto_front_val_norm[:, 1] = np.round(
            (self.opt_pareto_front_val_norm[:, 1] - self.min_MMACs) / (self.max_MMACs - self.min_MMACs), 4)
        self.opt_pareto_front_val_norm = np.round(self.opt_pareto_front_val_norm, 6)

        self.IGD_calc = IGD(self.opt_pareto_front_norm)
        self.IGD_s_calc = IGD(self.opt_pareto_front_val_norm)

        self.IGDp_calc = IGDPlus(self.opt_pareto_front_norm)
        self.IGDp_s_calc = IGDPlus(self.opt_pareto_front_val_norm)
        print('--> Set Up - Done')

    def _reset(self):
        self.test_performance_cache = {}
        self.training_based_performance_cache = {}
        self.training_free_performance_cache = {}
        self.efficiency_cache = {}

    def _get_a_compact_architecture(self):
        return np.random.choice(self.available_ops, self.maxLength)

    def get_test_performance(self, arch):
        hashKey = get_hashKey(arch, 'MacroNAS')
        try:
            test_error = self.test_performance_cache[hashKey]
        except KeyError:
            test_acc = self.api.get_test_accuracy(arch)
            test_error = np.round(1 - test_acc/100, 4)

            self.test_performance_cache[hashKey] = test_error
        return [test_error]

    def normalize_efficiency_metric(self, eff_metric):
        norm_eff_metric = (eff_metric - self.min_MMACs) / (self.max_MMACs - self.min_MMACs)
        norm_eff_metric = np.round(norm_eff_metric, 4)
        return norm_eff_metric

    def _get_val_performance_metric(self, arch, **kwargs):
        """
        - Get the validation performance of architecture.
        """
        hashKey = get_hashKey(arch, 'MacroNAS')
        indicator_time = 0.0
        evaluate_before = False
        try:
            performance_metric = self.training_based_performance_cache[hashKey]['performance_metric']
            benchmark_time = self.training_based_performance_cache[hashKey]['benchmark_time']
            evaluate_before = True
        except KeyError:
            val_acc, benchmark_time = self.api.get_val_accuracy(arch)
            performance_metric = np.round(1 - val_acc/100, 4)

            self.training_based_performance_cache[hashKey] = {}
            self.training_based_performance_cache[hashKey]['performance_metric'] = performance_metric
            self.training_based_performance_cache[hashKey]['benchmark_time'] = benchmark_time
        return [performance_metric], benchmark_time, indicator_time, evaluate_before

    def _get_efficiency_metric(self, arch, **kwargs):
        """
        - In the NAS-Bench-201 problem, the efficiency metric is nFLOPs.
        - It can be another metric such as number of trainable parameters, latency
        """
        hashKey = get_hashKey(arch, 'MacroNAS')
        try:
            nMMACs = self.efficiency_cache[hashKey]['nMMACs']
        except KeyError:
            nMMACs = np.round(self.api.get_efficiency_metrics(arch, metric='MMACs'), 6)

            self.efficiency_cache[hashKey] = {'MMACs': nMMACs}
        return [nMMACs]

    def _isValid(self, arch):
        return True

    def _calculate_IGD(self, approximation_front):
        approximation_front = np.array(approximation_front)
        approximation_front[:, 1] = (approximation_front[:, 1] - self.min_MMACs) / (self.max_MMACs - self.min_MMACs)
        approximation_front = np.round(approximation_front, 4)
        if self.opt_pareto_front is None:
            return -1
        IGD_value = self.IGD_calc(approximation_front)
        return IGD_value

    def _calculate_IGDp(self, approximation_front):
        approximation_front = np.array(approximation_front)
        approximation_front[:, 1] = (approximation_front[:, 1] - self.min_MMACs) / (self.max_MMACs - self.min_MMACs)
        approximation_front = np.round(approximation_front, 4)
        if self.opt_pareto_front is None:
            return -1
        IGDp_value = self.IGDp_calc(approximation_front)
        return IGDp_value

    def calculate_IGD_val(self, approximation_front):
        approximation_front = np.array(approximation_front)
        approximation_front[:, 1] = (approximation_front[:, 1] - self.min_MMACs) / (self.max_MMACs - self.min_MMACs)
        approximation_front = np.round(approximation_front, 4)
        if self.opt_pareto_front_val is None:
            return -1
        IGD_value = self.IGD_s_calc(approximation_front)
        return IGD_value

    def calculate_IGDp_val(self, approximation_front):
        approximation_front = np.array(approximation_front)
        approximation_front[:, 1] = (approximation_front[:, 1] - self.min_MMACs) / (self.max_MMACs - self.min_MMACs)
        approximation_front = np.round(approximation_front, 4)
        if self.opt_pareto_front_val is None:
            return -1
        IGDp_value = self.IGDp_s_calc(approximation_front)
        return IGDp_value

    def calculate_HV(self, approximation_front):
        approximation_front = np.array(approximation_front)
        approximation_front[:, 1] = (approximation_front[:, 1] - self.min_MMACs) / (self.max_MMACs - self.min_MMACs)
        approximation_front = np.round(approximation_front, 4)

        return np.round(HV_cal(approximation_front) / np.prod(reference_point), 6)

if __name__ == '__main__':
    pass
