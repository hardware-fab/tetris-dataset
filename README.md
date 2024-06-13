# TETRIS: A Dataset for FPGA-Based Heterogeneous Multi-Core Optimization

The design of heterogeneous SoCs has shown a growing trend in the latest
years, since the end of Moore's Law and Dennard Scaling made hardware
acceleration the only viable solution too keep the pace with increasing
performance demands in the HPC industry.
However, the state of the art lacks any general methodology for
accurately predicting the throughput of a hardware accelerator in a
system where resources like buses and memory are contended between many
computing elements.
A performance prediction is mandatory for SoC optimization, because both the
simulation and the FPGA implementation of complex SoCs require too much time
to explore a significative part of the design space.

The main idea behind TETRIS is quite simple: unleashing the power of
machine learning to predict the throughput of applications in different
configurations. TETRIS is a collection of area and performance results of
almost 5000 different SoC configurations, that differ for size and chosen
applications. This dataset aims to help researchers develop new methodologies
for effectively predicting the throughput of accelerators in environment with
high traffic and resource contention, boosting the research of new optimization
solutions for heterogeneous SoCs.

## Repository organization

The repository is organized as follows:

* `data` contains all the area and performance results related to the SoCs
  that have been implemented and tested. This folder is further divided in
  `UC` and `TC`.

* `models` includes the three models used to validate the TETRIS dataset,
  namely the Gaussian Process, the K-Nearest Neighbors, and the Random
  Forest.

* `scripts` contains two Python scripts, `regression_main.py` and
  `optimization_main.py`, that compute the
  regression error, and identify the optimal configuration with an iterative
  procedure given a set of constraints on the throughput, respectively.


## Getting started

The repository contains two Python scripts, `regression_main.py` and
`optimization_main.py`, intended to showcase two exemplifying usages of the
TETRIS dataset.

In order to use the scripts, the user must ensure the following Python
dependencies:

`pip install scikit_learn`

`pip install matplotlib`

Both the scripts can be launched by simply specifying the JSON file corresponding
to the scenario for which we would like to perform the regression or the optimization
(note that the optimization script is intended to be used with exhaustive scenarios,
i.e. the ones in the `UC` folder). For example, if we want to find the regression
error for the TC1 random scenario, from the main folder we run:

`python3 scripts/regression_main.py data/TC/tc1_rand.json  `

The scripts can be tuned to ajust the experiments as needed.
Regarding the `regression_main.py` script, the user can change the number of tests
and the explored training set sizes.
For the `optimization_main.py` script, the user can modify the number of tests,
the maximum number of iteration for the optimization process, and the throughput
threshold of each application (refer to the `tiles.json` file to get an idea of
the throughput of the various applications).


## Intended use

The goal of the TETRIS dataset is to allow researchers to develop methodologies
for performance prediction and optimization of complex SoCs without having to generate
the necessary hardware results by themselves.
In this sense, the scripts included in this repository can be viewed as templates for
the use and manipulation of the dataset.
A researcher wanting to design a novel methodology for performance prediction could start
by modifying the source files available in the `models` folder, developing new approaches that
adopt customized regression models and/or make use of a more extensive set of accelerators and
SoC features.



