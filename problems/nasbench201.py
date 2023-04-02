import numpy as np
import pickle as p
from api_benchmarks.api_201.api import API
from problems.NAS_problem import Problem
from utils import calculate_IGD_value, get_hashKey
from pymoo.indicators.hv import HV

# 'none': 0
# 'skip_connect': 1
# 'nor_conv_1x1': 2
# 'nor_conv_3x3': 3
# 'avg_pool_3x3': 4
reference_point = [1.01, 1.01]
HV_cal = HV(reference_point)

class NASBench201(Problem):
    def __init__(self, dataset, maxEvals, **kwargs):
        """
        # NAS-Benchmark-201 provides us the information (e.g., the training loss, the testing accuracy,
        the validation accuracy, the number of FLOPs, etc) of all architectures in the search space. Therefore, if we
        want to evaluate any architectures in the search space, we just need to query its information in the data.\n
        -----------------------------------------------------------------

        # Additional Hyper-parameters:\n
        - path_data -> the path contains NAS-Bench-201 data.
        - data -> NAS-Bench-201 data.
        - min_max -> the maximum and minimum of architecture's metrics. They are used to normalize the results.
        - opt_pareto_front -> the Pareto-optimal front in the search space (nFLOPs - testing error)
        - available_ops -> the available operators can choose in the search space.
        - maxLength -> the maximum length of compact architecture.
        """

        super().__init__(maxEvals, 'NASBench201', dataset, **kwargs)
        self.objective_0 = 'Test Error'
        self.objective_1 = '#FLOPs'

        self.available_ops = [0, 1, 2, 3, 4]
        self.maxLength = 6

        self.data_path = kwargs['path_api_benchmark'] + f'/NASBench201'
        self.pareto_opt_front_path = kwargs['path_pareto_optimal_front']

        self.min_FLOPs, self.max_FLOPs = None, None

        self.opt_pareto_front = None
        self.tf_estimator = None

        self.test_performance_cache = {}
        self.training_based_performance_cache = {}
        self.efficiency_cache = {}

    def _set_up(self):
        available_subsets = ['CIFAR-10', 'CIFAR-100', 'ImageNet16-120']
        if self.dataset not in available_subsets:
            raise ValueError(f'Just only supported these subsets: CIFAR-10; CIFAR-100; ImageNet16-120.'
                             f'{self.dataset} subset is not supported at this time.')

        self.api = API(data_path=self.data_path, dataset=self.dataset)

        if self.dataset == 'CIFAR-10':
            self.min_FLOPs, self.max_FLOPs = 7.78305, 220.11969
        elif self.dataset == 'CIFAR-100':
            self.min_FLOPs, self.max_FLOPs = 7.78890, 220.12554
        else:
            self.min_FLOPs, self.max_FLOPs = 1.95340, 55.03756

        f_opt_pareto_front = open(f'{self.pareto_opt_front_path}/[POF_TestAcc_FLOPs]_[NAS201_{self.dataset}].p', 'rb')
        self.opt_pareto_front = p.load(f_opt_pareto_front)
        self.opt_pareto_front[:, 0] /= 100
        f_opt_pareto_front.close()

        self.opt_pareto_front_norm = self.opt_pareto_front.copy()
        self.opt_pareto_front_norm[:, 1] = np.round(
            (self.opt_pareto_front_norm[:, 1] - self.min_FLOPs) / (self.max_FLOPs - self.min_FLOPs), 4)
        self.opt_pareto_front_norm = np.round(self.opt_pareto_front_norm, 6)

        f_opt_pareto_front_val = open(f'{self.pareto_opt_front_path}/[POF_ValAcc_FLOPs]_[NAS201_{self.dataset}].p', 'rb')
        self.opt_pareto_front_val = p.load(f_opt_pareto_front_val)
        self.opt_pareto_front_val[:, 0] /= 100
        f_opt_pareto_front_val.close()

        self.opt_pareto_front_val_norm = self.opt_pareto_front_val.copy()
        self.opt_pareto_front_val_norm[:, 1] = np.round(
            (self.opt_pareto_front_val_norm[:, 1] - self.min_FLOPs) / (self.max_FLOPs - self.min_FLOPs), 4)
        self.opt_pareto_front_val_norm = np.round(self.opt_pareto_front_val_norm, 6)

        print('--> Set Up - Done')

    def _reset(self):
        self.test_performance_cache = {}
        self.training_based_performance_cache = {}
        self.efficiency_cache = {}

    def _get_a_compact_architecture(self):
        return np.random.choice(self.available_ops, self.maxLength)

    def get_test_performance(self, arch):
        hashKey = get_hashKey(arch, 'NASBench201')
        try:
            test_error = self.test_performance_cache[hashKey]
        except KeyError:
            test_acc = self.api.get_test_accuracy(arch)
            test_error = np.round(1 - test_acc, 4)

            self.test_performance_cache[hashKey] = test_error
        return [test_error]

    def normalize_efficiency_metric(self, eff_metric):
        norm_eff_metric = (eff_metric - self.min_FLOPs) / (self.max_FLOPs - self.min_FLOPs)
        norm_eff_metric = np.round(norm_eff_metric, 4)
        return norm_eff_metric

    def _get_val_performance_metric(self, arch, **kwargs):
        """
        - Get the validation performance of architecture.
        """
        hashKey = get_hashKey(arch, 'NASBench201')
        indicator_time = 0.0
        evaluate_before = False
        try:
            performance_metric = self.training_based_performance_cache[hashKey]['performance_metric']
            benchmark_time = self.training_based_performance_cache[hashKey]['benchmark_time']
            evaluate_before = True
        except KeyError:
            try:
                epoch = kwargs['epoch']
            except KeyError:
                raise ValueError('Do not set the epoch to query the performance.')

            val_acc, benchmark_time = self.api.get_val_accuracy(arch, iepoch=epoch)
            performance_metric = np.round(1 - val_acc, 4)

            self.training_based_performance_cache[hashKey] = {}
            self.training_based_performance_cache[hashKey]['performance_metric'] = performance_metric
            self.training_based_performance_cache[hashKey]['benchmark_time'] = benchmark_time
        return [performance_metric], benchmark_time, indicator_time, evaluate_before

    def _get_efficiency_metric(self, arch, **kwargs):
        """
        - In the NAS-Bench-201 problem, the efficiency metric is nFLOPs.
        - It can be another metric such as number of trainable parameters, latency
        """
        hashKey = get_hashKey(arch, 'NASBench201')
        try:
            nFLOPs = self.efficiency_cache[hashKey]['FLOPs']
        except KeyError:
            nFLOPs = np.round(self.api.get_efficiency_metrics(arch, metric='FLOPs'), 6)

            self.efficiency_cache[hashKey] = {'FLOPs': nFLOPs}
        return [nFLOPs]

    def _isValid(self, arch):
        return True

    def _calculate_IGD(self, approximation_front):
        approximation_front = np.array(approximation_front)
        approximation_front[:, 1] = self.normalize_efficiency_metric(approximation_front[:, 1])
        approximation_front = np.round(approximation_front, 4)
        if self.opt_pareto_front is None:
            return -1
        IGD = calculate_IGD_value(pareto_optimal_front=self.opt_pareto_front_norm, approximation_front=approximation_front)
        return IGD

    def calculate_IGD_val(self, approximation_front):
        approximation_front = np.array(approximation_front)
        approximation_front[:, 1] = self.normalize_efficiency_metric(approximation_front[:, 1])
        approximation_front = np.round(approximation_front, 4)
        if self.opt_pareto_front_val is None:
            return -1
        IGD = calculate_IGD_value(pareto_optimal_front=self.opt_pareto_front_val_norm, approximation_front=approximation_front)
        return IGD

    def calculate_HV(self, approximation_front):
        approximation_front = np.array(approximation_front)
        approximation_front[:, 1] = self.normalize_efficiency_metric(approximation_front[:, 1])
        approximation_front = np.round(approximation_front, 4)

        return np.round(HV_cal(approximation_front) / np.prod(reference_point), 6)

if __name__ == '__main__':
    pass
