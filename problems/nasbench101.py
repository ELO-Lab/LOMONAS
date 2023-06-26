import numpy as np
import pickle as p
from problems.NAS_problem import Problem
from utils import get_hashKey
from pymoo.indicators.hv import HV
from pymoo.indicators.igd import IGD
from pymoo.indicators.igd_plus import IGDPlus

from api_benchmarks.api_101 import wrap_api as api

# 1: INPUT
# 2: CONV 1x1
# 3: CONV 3x3
# 4: MAXPOOL 3x3
# 5: OUTPUT

reference_point = [1.01, 1.01]
HV_cal = HV(reference_point)

class NASBench101(Problem):
    def __init__(self, max_eval, dataset='CIFAR-10', **kwargs):
        """
        # NAS-Benchmark-101 provides us the information (e.g., the testing accuracy, the validation accuracy,
        the number of parameters) of all architectures in the search space. Therefore, if we want to evaluate any
        architectures in the search space, we just need to query its information in the data.\n
        -----------------------------------------------------------------

        - path_data -> the path contains NAS-Bench-101 data.
        - data -> NAS-Bench-101 data.
        - opt_pareto_front -> the Pareto-optimal front in the search space (nFLOPs - testing error)
        - OPS -> the available operators can choose in the search space.
        - IDX_OPS -> the index of operators in compact architecture.
        - EDGES -> 0: doesn't have edge; 1: have edge.
        - IDX_EDGES -> the index of edges in compact architecture.
        - maxLength -> the maximum length of compact architecture.
        """

        super().__init__(max_eval, 'NASBench101', dataset, **kwargs)

        self.objective_0 = 'Test Error'
        self.objective_1 = '#Params'

        self.OPS = [2, 3, 4]
        self.IDX_OPS = [1, 3, 6, 10, 15]

        self.EDGES = [0, 1]
        self.IDX_EDGES = [2, 4, 5, 7, 8, 9, 11, 12, 13, 14, 16, 17, 18, 19, 20, 22, 23, 24, 25, 26, 27]

        self.maxLength = 28

        self.path_data = kwargs['api_benchmark_path'] + '/NASBench101'
        self.pof_path = kwargs['pof_path']

        self.min_params, self.max_params = None, None

        self.opt_pareto_front = None
        self.tf_estimator = None

        self.test_performance_cache = {}
        self.training_based_performance_cache = {}
        self.training_free_performance_cache = {}
        self.efficiency_cache = {}

    def _set_up(self):
        self.api = api.NASBench_(self.path_data + '/data.p')

        self.min_params, self.max_params = 227274, 49979274

        f_opt_pareto_front = open(f'{self.pof_path}/[POF_TestAcc_Params]_[NAS101].p', 'rb')
        self.opt_pareto_front = p.load(f_opt_pareto_front)
        f_opt_pareto_front.close()

        self.opt_pareto_front_norm = self.opt_pareto_front.copy()
        self.opt_pareto_front_norm[:, 1] = np.round((self.opt_pareto_front_norm[:, 1] - self.min_params) / (self.max_params - self.min_params), 4)
        self.opt_pareto_front_norm = np.round(self.opt_pareto_front_norm, 6)

        f_opt_pareto_front_val = open(f'{self.pof_path}/[POF_ValAcc_Params]_[NAS101].p', 'rb')
        self.opt_pareto_front_val = p.load(f_opt_pareto_front_val)
        f_opt_pareto_front_val.close()

        self.opt_pareto_front_val_norm = self.opt_pareto_front_val.copy()
        self.opt_pareto_front_val_norm[:, 1] = np.round(
            (self.opt_pareto_front_val_norm[:, 1] - self.min_params) / (self.max_params - self.min_params), 4)
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
        arch[self.IDX_OPS] = np.random.choice(self.OPS, len(self.IDX_OPS))
        arch[self.IDX_EDGES] = np.random.choice(self.EDGES, len(self.IDX_EDGES))
        arch[0] = 1
        arch[21] = 5
        return arch

    """--------------------------------------------- Performance Metrics --------------------------------------------"""
    def get_test_performance(self, arch):
        hashKey = get_hashKey(arch, 'NASBench101')
        try:
            test_error = self.test_performance_cache[hashKey]
        except KeyError:
            test_acc = self.api.get_test_accuracy(arch)
            test_error = np.round(1 - test_acc, 4)

            self.test_performance_cache[hashKey] = test_error
        return [test_error]

    def normalize_efficiency_metric(self, eff_metric):
        norm_eff_metric = (eff_metric - self.min_params) / (self.max_params - self.min_params)
        norm_eff_metric = np.round(norm_eff_metric, 4)
        return norm_eff_metric

    def _get_val_performance_metric(self, arch, **kwargs):
        """
        - Get the validation performance of architecture.
        """
        hashKey = get_hashKey(arch, 'NASBench101')
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
            val_acc, benchmark_time = self.api.get_val_accuracy(arch, epoch=epoch)
            performance_metric = np.round(1 - val_acc, 4)

            self.training_based_performance_cache[hashKey] = {}
            self.training_based_performance_cache[hashKey]['performance_metric'] = performance_metric
            self.training_based_performance_cache[hashKey]['benchmark_time'] = benchmark_time
        return [performance_metric], benchmark_time, indicator_time, evaluate_before

    def _get_efficiency_metric(self, arch, **kwargs):
        """
        - In the NAS-Bench-101 problem, the efficiency metric is the number of trainable parameters
        """
        hashKey = get_hashKey(arch, 'NASBench101')
        try:
            nParams = self.efficiency_cache[hashKey]['nParams']
        except KeyError:
            nParams = np.round(self.api.get_efficiency_metrics(arch, metric='params'), 6)
            self.efficiency_cache[hashKey] = {'nParams': nParams}
        return [nParams]

    def _isValid(self, X):
        edges_matrix, ops_matrix = api.X_2_matrices(X)
        model_spec = api.ModelSpec(edges_matrix, ops_matrix)
        return self.api.is_valid(model_spec)

    def _calculate_IGD(self, approximation_front):
        approximation_front = np.array(approximation_front)
        approximation_front[:, 1] = (approximation_front[:, 1] - self.min_params) / (self.max_params - self.min_params)
        approximation_front = np.round(approximation_front, 4)
        if self.opt_pareto_front is None:
            return -1
        # IGD_value = calculate_IGD_value(pareto_optimal_front=self.opt_pareto_front_norm, approximation_front=approximation_front)
        IGD_value = self.IGD_calc(approximation_front)
        return IGD_value

    def _calculate_IGDp(self, approximation_front):
        approximation_front = np.array(approximation_front)
        approximation_front[:, 1] = (approximation_front[:, 1] - self.min_params) / (self.max_params - self.min_params)
        approximation_front = np.round(approximation_front, 4)
        if self.opt_pareto_front is None:
            return -1
        IGDp_value = self.IGDp_calc(approximation_front)
        return IGDp_value

    def calculate_IGD_val(self, approximation_front):
        approximation_front = np.array(approximation_front)
        approximation_front[:, 1] = (approximation_front[:, 1] - self.min_params) / (self.max_params - self.min_params)
        approximation_front = np.round(approximation_front, 4)
        if self.opt_pareto_front_val is None:
            return -1
        # IGD_value = calculate_IGD_value(pareto_optimal_front=self.opt_pareto_front_val_norm, approximation_front=approximation_front)
        IGD_value = self.IGD_s_calc(approximation_front)
        return IGD_value

    def calculate_IGDp_val(self, approximation_front):
        approximation_front = np.array(approximation_front)
        approximation_front[:, 1] = (approximation_front[:, 1] - self.min_params) / (self.max_params - self.min_params)
        approximation_front = np.round(approximation_front, 4)
        if self.opt_pareto_front_val is None:
            return -1
        IGDp_value = self.IGDp_s_calc(approximation_front)
        return IGDp_value

    def calculate_HV(self, approximation_front):
        approximation_front = np.array(approximation_front)
        approximation_front[:, 1] = (approximation_front[:, 1] - self.min_params) / (self.max_params - self.min_params)
        approximation_front = np.round(approximation_front, 4)

        return np.round(HV_cal(approximation_front) / np.prod(reference_point), 6)

if __name__ == '__main__':
    pass
