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
## Search
### Evolution-based MONAS (NSGA-II & MOEA/D)
```shell
$ python run_EMONAS.py
--optimizer [MOEA_NSGAII, MOEA_MOEAD] <search_strategy>
--pop_size <population_size, default: 20>
--problem [NAS101, NAS201-C10, NAS201-C100, NAS201-IN16, MacroNAS-C10, MacroNAS-C100, NAS-ASR] <experiment_problem>
--maxEvals <maximum_number_of_evaluations, default: 3000>
--seed <initial_random_seed, default: 0>
--n_runs <number_of_experiment_runs, default: 31>
--path_api_benchmark <path_of_benchmark_databases, default: ./data>
--path_results <path_of_experimental_results, default: ./exp_rs>
```
### LOMONAS & Random Restart Local Search (RR-LS)
```shell
$ python run_MOLS.py
--optimizer [LOMONAS, RR_LS] <search_strategy>
--NF <number_of_kept_fronts (for LOMONAS only), default: 3>
--get_all_neighbors <check_all_neighbors? (for LOMONAS only), default: 0>
--local_search_on_all_sols <perform_local_search_on_all_solutions? (for LOMONAS only)', default: 0>
--loop <RR-LS_with_loop? (for RR-LS only)', default: 0>
--problem [NAS101, NAS201-C10, NAS201-C100, NAS201-IN16, MacroNAS-C10, MacroNAS-C100, NAS-ASR] <experiment_problem>
--maxEvals <maximum_number_of_evaluations, default: 3000>
--seed <initial_random_seed, default: 0>
--n_runs <number_of_experiment_runs, default: 31>
--path_api_benchmark <path_of_benchmark_databases, default: ./data>
--path_results <path_of_experimental_results, default: ./exp_rs>
```
The configurations of LOMONAS proposed in paper are ```--NF 3```, ```--get_all_neighbors 0```, and ```--local_search_on_all_sols 0```.

Example script of performing LOMONAS on NAS-Bench-201 search space (CIFAR-100):
```shell
$ python run_MOLS.py --optimizer LOMONAS --NF 3 --get_all_neighbors 0 --local_search_on_all_sols 0 --problem NAS201-C100
```
## Transferability Evaluation
```shell
$ python run_transfer.py
--problem [NAS201-C100, NAS201-IN16, MacroNAS-C100] <experiment_problem>
--path_api_benchmark <path_of_benchmark_databases, default: ./data>
--path_pre_results <path_of_previous_experimental_results>
```
In our study, we evaluate the transferability of algorithms by evaluating the search results (on CIFAR-10) on CIFAR-100 and ImageNet16-120 (for NAS-Bench-201 only).

```--path_pre_results``` must contain the search results at NAS201-C10 or MacroNAS-C10 problems.

## Acknowledgement
Our source code is inspired by:
- [pymoo: Multi-objective Optimization in Python](https://github.com/anyoptimization/pymoo)
- [NSGA-Net: Neural Architecture Search using Multi-Objective Genetic Algorithm](https://github.com/ianwhale/nsga-net)
- [NAS-Bench-101: Towards Reproducible Neural Architecture Search](https://github.com/google-research/nasbench)
- [NAS-Bench-201: Extending the Scope of Reproducible Neural Architecture Search](https://github.com/D-X-Y/NAS-Bench-201)
- [NAS-Bench-ASR: Reproducible Neural Architecture Search for Speech Recognition](https://github.com/AbhinavMehrotra/nb-asr)
- [Local Search is a Remarkably Strong Baseline for Neural Architecture Search](https://github.com/tdenottelander/MacroNAS)
