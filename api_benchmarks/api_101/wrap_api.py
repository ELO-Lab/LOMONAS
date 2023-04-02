from api_benchmarks.api_101.api import NASBench
from api_benchmarks.api_101.lib import model_spec as _model_spec
import numpy as np

ModelSpec = _model_spec.ModelSpec

idx_ops = [1, 3, 6, 10, 15]
idx_edges = [2, 4, 5, 7, 8, 9, 11, 12, 13, 14, 16, 17, 18, 19, 20, 22, 23, 24, 25, 26, 27]


def X_2_matrices(X):
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
            if X[edge] - 1 == 0:
                edges_matrix[row][row + i + 1] = 1

    ops_matrix = ['input']
    for i in idx_ops:
        if X[i] == 2:
            ops_matrix.append('conv1x1-bn-relu')
        elif X[i] == 3:
            ops_matrix.append('conv3x3-bn-relu')
        else:
            ops_matrix.append('maxpool3x3')
    ops_matrix.append('output')

    return edges_matrix, ops_matrix


class OutOfDomainError(Exception):
    """Indicates that the requested graph is outside of the search domain."""


class NASBench_(NASBench):
    def __init__(self, dataset_file):
        super().__init__(dataset_file=dataset_file)

    def get_module_hash(self, model_spec):
        return self._hash_spec(model_spec)

    def query(self, model_spec, epochs=108, stop_halfway=False):
        """Fetch one of the evaluations for this model spec.

            Each call will sample one of the config['num_repeats'] evaluations of the
            model. This means that repeated queries of the same model (or isomorphic
            models) may return identical metrics.

            This function will increment the budget counters for benchmarking purposes.
            See self.training_time_spent, and self.total_epochs_spent.

            This function also allows querying the evaluation metrics at the halfway
            point of training using stop_halfway. Using this option will increment the
            budget counters only up to the halfway point.

            Args:
              model_spec: ModelSpec object.
              epochs: number of epochs trained. Must be one of the evaluated number of
                epochs, [4, 12, 36, 108] for the full dataset.
              stop_halfway: if True, returned dict will only contain the training time
                and accuracies at the halfway point of training (num_epochs/2).
                Otherwise, returns the time and accuracies at the end of training
                (num_epochs).

            Returns:
              dict containing the evaluated data for this object.

            Raises:
              OutOfDomainError: if model_spec or num_epochs is outside the search space.
            """
        fixed_stat, computed_stat = self.get_metrics_from_spec(model_spec)

        computed_stat_0 = computed_stat[epochs][0]
        computed_stat_1 = computed_stat[epochs][1]
        computed_stat_2 = computed_stat[epochs][2]

        data = dict()
        data['module_adjacency'] = fixed_stat['module_adjacency']
        data['module_operations'] = fixed_stat['module_operations']
        data['trainable_parameters'] = fixed_stat['trainable_parameters']

        if stop_halfway:
            data['training_time'] = computed_stat['halfway_training_time']
            data['train_accuracy'] = computed_stat['halfway_train_accuracy']
            data['validation_accuracy'] = computed_stat['halfway_validation_accuracy']
            data['test_accuracy'] = computed_stat['halfway_test_accuracy']
        else:
            data['training_time'] = computed_stat_0['final_training_time']
            data['training_time'] += computed_stat_1['final_training_time']
            data['training_time'] += computed_stat_2['final_training_time']
            data['training_time'] /= 3

            data['train_accuracy'] = computed_stat_0['final_train_accuracy']
            data['train_accuracy'] += computed_stat_1['final_train_accuracy']
            data['train_accuracy'] += computed_stat_2['final_train_accuracy']
            data['train_accuracy'] /= 3

            data['validation_accuracy'] = computed_stat_0['final_validation_accuracy']
            data['validation_accuracy'] += computed_stat_1['final_validation_accuracy']
            data['validation_accuracy'] += computed_stat_2['final_validation_accuracy']
            data['validation_accuracy'] /= 3

            data['test_accuracy'] = computed_stat_0['final_test_accuracy']
            data['test_accuracy'] += computed_stat_1['final_test_accuracy']
            data['test_accuracy'] += computed_stat_2['final_test_accuracy']
            data['test_accuracy'] /= 3

        self.training_time_spent += data['training_time']
        if stop_halfway:
            self.total_epochs_spent += epochs // 2
        else:
            self.total_epochs_spent += epochs

        return data

    def arch_2_key(self, arch):
        edges_matrix, ops_matrix = X_2_matrices(arch)
        model_spec = ModelSpec(edges_matrix, ops_matrix)
        key = self.get_module_hash(model_spec)
        return key

    def get_test_accuracy(self, arch, epoch=108):
        keyData = self.arch_2_key(arch)
        return self.data[f'{epoch}'][keyData]['test_acc']

    def get_training_time(self, keyData, epoch):
        return self.data[f'{epoch}'][keyData]['train_time']

    def get_val_accuracy(self, arch, epoch=12):
        keyData = self.arch_2_key(arch)
        return self.data[f'{epoch}'][keyData]['val_acc'], self.get_training_time(keyData=keyData, epoch=epoch)

    def get_efficiency_metrics(self, arch, metric='params'):
        keyData = self.arch_2_key(arch)
        efficiency_metrics = {
            'params': self.data['108'][keyData]['n_params'],
        }
        return efficiency_metrics[metric]