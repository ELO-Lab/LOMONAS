# LOMONAS: Local-search algorithm for Multi-Objective Neural Architecture Search
[![MIT licensed](https://img.shields.io/badge/license-MIT-brightgreen.svg)](LICENSE.md)

Quan Minh Phan, Ngoc Hoang Luong.

In GECCO 2023.
## Setup
- Clone repo.
- Install necessary packages.
```
$ pip install -r requirements.txt
```
-  Download databases in this [link](https://drive.google.com/drive/folders/1jAX-By0UUOld_vLRLBLX1GppQ6lhcOvS?usp=sharing), unzip and put all folders into ```data``` folder for building APIs (benchmarks).

In our experiments, we do not implement directly the API benchmarks published in their repos (e.g., NAS-Bench-101, NAS-Bench-201, etc).
Instead, we create smaller-size databases by accessing their databases and only logging necessary content.

You can compare our databases and the original databases in [check_log_database.ipynb](check_log_database.ipynb)
## Reproducing the results
You can reproduce our results by running the below script:
```shell
$ python main.py --optimizer [MOEA_NSGAII, MOEA_MOEAD, RR_LS, LOMONAS] --problem [NAS101, MacroNAS-C10, MacroNAS-C100, NAS201-C10, NAS201-C100, NAS201-IN16, NAS-ASR]
```
## Search with different hyperparameters
Moreover, you can search with different hyperparameter settings
### For environment
- `n_run`: the number times of running algorithms (default: `31`)
- `max_eval`: the maximum number of evaluation each run (default: `3000`)
- `init_seed`: the initial random seed (default: `0`)
- `api_benchmark_path`: the path that contains the api databases (default: `./data`)
- `res_path`: the path for logging results (default: `./exp_res`)
- `debug`: print the search performance at each generation if `debug` is `True` (default: `False`)
### For evolution-based MONAS (NSGA-II & MOEA/D)
- `pop_size`: the population size
### For [Random Restart - Local Search](https://github.com/tdenottelander/MacroNAS)
- `loop`: implement the loop RR-LS variant if `loop` is `True` (default: `False`)
### For LOMONAS
- `NF`: the number of fronts kept for performing neighborhood checks (default: `3`)
- `check_all_neighbors`: evaluate all solutions in the neighborhood if set `True` (default: `False`)
- `neighborhood_check_on_all_sols`: perform neighborhood check on all solutions instead of knee and extremes ones if set `True` (default: `False`)

The results of LOMONAS in the paper are run with the default hyperparameters.

## Transferability Evaluation
In our study, we evaluate the transferability of algorithms by evaluating the found architectures (search on CIFAR-10) on CIFAR-100 (for MacroNAS and NAS-201) and ImageNet16-120 (for NAS-201 only).

```shell
$ python transferability_evaluation.py
--problem [MacroNAS-C100, NAS201-C100, NAS201-IN16]
--api_benchmark_path <path_of_benchmark_databases, default: ./data>
--cifar10_res_path
```
Note: ```--cifar10_res_path``` must contain the searching result at MacroNAS-C10 or NAS201-C10 problems.

For example:
```shell
$ python transferability_evaluation.py --problem NAS201-C100 --cifar10_res_path ./exp_res/NAS201-C10 --api_benchmark_path ./data
```
## Visualization and T-test
You can create all figures presented in our paper by run the [visualization_and_statistical_test.ipynb](visualization_and_statistical_test.ipynb) file.

## Acknowledgement
Our source code is inspired by:
- [pymoo: Multi-objective Optimization in Python](https://github.com/anyoptimization/pymoo)
- [NSGA-Net: Neural Architecture Search using Multi-Objective Genetic Algorithm](https://github.com/ianwhale/nsga-net)
- [NAS-Bench-101: Towards Reproducible Neural Architecture Search](https://github.com/google-research/nasbench)
- [NAS-Bench-201: Extending the Scope of Reproducible Neural Architecture Search](https://github.com/D-X-Y/NAS-Bench-201)
- [NAS-Bench-ASR: Reproducible Neural Architecture Search for Speech Recognition](https://github.com/AbhinavMehrotra/nb-asr)
- [Local Search is a Remarkably Strong Baseline for Neural Architecture Search](https://github.com/tdenottelander/MacroNAS)
