import numpy as np
import pickle as p
from api_benchmarks.api_ASR.api import API
from problems.NAS_problem import Problem
from utils import calculate_IGD_value, get_hashKey
from pymoo.indicators.hv import HV
from pymoo.indicators.igd import IGD
from pymoo.indicators.igd_plus import IGDPlus

# 'linear': 0
# '1x5 Conv': 1
# '1x5 Conv, dil 2': 2
# '1x7 Conv': 3
# '1x7 Conv, dil 2': 4
# 'zeroize': 5

# 'identity': 0
# 'zeroize': 1
reference_point = [1.01, 1.01]
HV_cal = HV(reference_point)

class NASBenchASR(Problem):
    def __init__(self, dataset, max_eval, **kwargs):
        """
        # NAS-Benchmark-ASR provides us the information (e.g., the testing PER, the validation PER, the number of FLOPs,
        etc) of all architectures in the search space.
        Therefore, if we want to evaluate any architectures in the search space, we just need to query its information in the data.\n
        -----------------------------------------------------------------

        # Additional Hyper-parameters:\n
        - path_data -> the path contains NAS-Bench-ASR data.
        - data -> NAS-Bench-ASR data.
        - opt_pareto_front -> the Pareto-optimal front in the search space (testing PER - nFLOPs)
        - maxLength -> the maximum length of compact architecture.
        """

        super().__init__(max_eval, 'NASBenchASR', dataset, **kwargs)
        self.objective_0 = 'test_per'
        self.objective_1 = '#FLOPs'

        self.MAIN_OPS = [0, 1, 2, 3, 4, 5]
        self.IDX_MAIN_OPS = [0, 2, 5]

        self.SKIP_OPS = [0, 1]
        self.IDX_SKIP_OPS = [1, 3, 4, 6, 7, 8]

        self.maxLength = 9

        self.data_path = kwargs['path_api_benchmark'] + f'/NASBenchASR'
        self.pareto_opt_front_path = kwargs['path_pareto_optimal_front']

        self.min_FLOPs, self.max_FLOPs = None, None

        self.opt_pareto_front = None
        self.tf_estimator = None

        self.test_performance_cache = {}
        self.training_based_performance_cache = {}
        self.training_free_performance_cache = {}
        self.efficiency_cache = {}

    def _set_up(self):
        self.api = API(data_path=self.data_path, dataset=self.dataset)

        self.min_FLOPs, self.max_FLOPs = 1982027266, 6968537266

        f_opt_pareto_front = open(f'{self.pareto_opt_front_path}/[POF_TestPER_FLOPs]_[NAS-ASR].p', 'rb')
        self.opt_pareto_front = p.load(f_opt_pareto_front)
        f_opt_pareto_front.close()

        self.opt_pareto_front_norm = self.opt_pareto_front.copy()
        self.opt_pareto_front_norm[:, 1] = np.round(
            (self.opt_pareto_front_norm[:, 1] - self.min_FLOPs) / (self.max_FLOPs - self.min_FLOPs), 4)
        self.opt_pareto_front_norm = np.round(self.opt_pareto_front_norm, 6)

        f_opt_pareto_front_val = open(f'{self.pareto_opt_front_path}/[POF_ValPER_FLOPs]_[NAS-ASR].p', 'rb')
        self.opt_pareto_front_val = p.load(f_opt_pareto_front_val)
        f_opt_pareto_front.close()

        self.opt_pareto_front_val_norm = self.opt_pareto_front_val.copy()
        self.opt_pareto_front_val_norm[:, 1] = np.round(
            (self.opt_pareto_front_val_norm[:, 1] - self.min_FLOPs) / (self.max_FLOPs - self.min_FLOPs), 4)
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

    def set_training_free_estimator(self, tf_estimator):
        self.tf_estimator = tf_estimator

    def _get_a_compact_architecture(self):
        arch = np.zeros(self.maxLength, dtype=np.int8)
        arch[self.IDX_MAIN_OPS] = np.random.choice(self.MAIN_OPS, len(self.IDX_MAIN_OPS))
        arch[self.IDX_SKIP_OPS] = np.random.choice(self.SKIP_OPS, len(self.IDX_SKIP_OPS))
        return arch

    def get_test_performance(self, arch):
        hashKey = get_hashKey(arch, 'NASBenchASR')
        try:
            test_per = self.test_performance_cache[hashKey]
        except KeyError:
            test_per = self.api.get_test_per(arch)

            self.test_performance_cache[hashKey] = test_per
        return [test_per]

    def get_training_free_performance_metric(self, arch):
        pass

    def normalize_efficiency_metric(self, eff_metric):
        norm_eff_metric = (eff_metric - self.min_FLOPs) / (self.max_FLOPs - self.min_FLOPs)
        norm_eff_metric = np.round(norm_eff_metric, 4)
        return norm_eff_metric

    def _get_val_performance_metric(self, arch, **kwargs):
        """
        - Get the validation performance of architecture.
        """
        hashKey = get_hashKey(arch, 'NASBenchASR')
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

            val_per, benchmark_time = self.api.get_val_per(arch, iepoch=epoch)
            performance_metric = np.round(val_per, 4)

            self.training_based_performance_cache[hashKey] = {}
            self.training_based_performance_cache[hashKey]['performance_metric'] = performance_metric
            self.training_based_performance_cache[hashKey]['benchmark_time'] = benchmark_time
        return [performance_metric], benchmark_time, indicator_time, evaluate_before

    def _get_efficiency_metric(self, arch, **kwargs):
        """
        - In the NAS-Bench-201 problem, the efficiency metric is nFLOPs.
        - It can be another metric such as number of trainable parameters, latency
        """
        hashKey = get_hashKey(arch, 'NASBenchASR')
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
        approximation_front[:, 1] = (approximation_front[:, 1] - self.min_FLOPs) / (self.max_FLOPs - self.min_FLOPs)
        approximation_front = np.round(approximation_front, 4)
        if self.opt_pareto_front is None:
            return -1
        IGD_value = self.IGD_calc(approximation_front)
        return IGD_value

    def _calculate_IGDp(self, approximation_front):
        approximation_front = np.array(approximation_front)
        approximation_front[:, 1] = (approximation_front[:, 1] - self.min_FLOPs) / (self.max_FLOPs - self.min_FLOPs)
        approximation_front = np.round(approximation_front, 4)
        if self.opt_pareto_front is None:
            return -1
        IGDp_value = self.IGDp_calc(approximation_front)
        return IGDp_value

    def calculate_IGD_val(self, approximation_front):
        approximation_front = np.array(approximation_front)
        approximation_front[:, 1] = (approximation_front[:, 1] - self.min_FLOPs) / (self.max_FLOPs - self.min_FLOPs)
        approximation_front = np.round(approximation_front, 4)
        if self.opt_pareto_front_val is None:
            return -1
        IGD_value = self.IGD_s_calc(approximation_front)
        return IGD_value

    def calculate_IGDp_val(self, approximation_front):
        approximation_front = np.array(approximation_front)
        approximation_front[:, 1] = (approximation_front[:, 1] - self.min_FLOPs) / (self.max_FLOPs - self.min_FLOPs)
        approximation_front = np.round(approximation_front, 4)
        if self.opt_pareto_front_val is None:
            return -1
        IGDp_value = self.IGDp_s_calc(approximation_front)
        return IGDp_value

    def calculate_HV(self, approximation_front):
        approximation_front = np.array(approximation_front)
        approximation_front[:, 1] = (approximation_front[:, 1] - self.min_FLOPs) / (self.max_FLOPs - self.min_FLOPs)
        approximation_front = np.round(approximation_front, 4)

        return np.round(HV_cal(approximation_front) / np.prod(reference_point), 6)

if __name__ == '__main__':
    pass
