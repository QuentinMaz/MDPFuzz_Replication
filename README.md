# Replicate of MDPFuzz
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

This repository contains a re-implementation the policy testing framework [MDPFuzz](https://sites.google.com/view/mdpfuzz/evaluation-results?authuser=0), whose original code can be found [here](https://github.com/Qi-Pang/MDPFuzz).
This tool has been used in the paper *Replicability Study: Policy Testing with MDPFuzz*.

## Installation

You can install the package with `pip`:
```
pip install mdpfuzz
```
One can also want to install the package locally:
```
git clone https://github.com/QuentinMaz/MDPFuzz_Replication mdpfuzz
cd mdpfuzz
pip install -e .
```
In the latter case, don't forget to append the folder `mdpfuzz/` to your path.

## Usage

The *Fuzzer* class (`mdpfuzz/mdpfuzz.py`) provides functions for fuzzing with and without GMM guidance, as well as a simple random testing procedure.
It inputs an *Executor* object, which is responsible for generating and mutating inputs (also called *seeds*), loading the policy under test and executing test cases (i.e., running the policy with a given input).
As such, using the package involves 3 simple steps:
1. Implementing a *Executor* class for your use case.
2. Creating an *executor*, loading the model under test and instantiate a *fuzzer*.
3. Running the *fuzzer* (fuzzing - with or without GMM guidance - or Random Testing) with the testing budget you want!

### Example

In the file `dqn_mountain_car.py`, we test a DQN agent (learnt using the library [Stable Baselines3](https://github.com/DLR-RM/stable-baselines3)) for the [Mountain Car](https://gymnasium.farama.org/environments/classic_control/mountain_car/) environment.

#### Setup

Setup a virtual environment with, for instance, [conda](https://docs.conda.io/en/latest/):
```bash
conda create -n demo python=3.10.12
conda activate demo
pip install mdpfuzz
# install gymnasium and stable-baselines3 dependencies
pip install gymnasium==0.29.1
pip install stable-baselines3==2.2.1
```
Make sure to activate the virtual environment (`conda activate demo`) before executing the script!

Additionally, we provide the script as a [Jupyter notebook](https://jupyter.org/) in which with we detail each step of the demonstration.
If you use the notebook, install the following dependency in the virtual environment: `pip install ipykernel`.

The demonstration consists of:
1. Learning and saving the DQN agent (`dqn_mountain_car.zip`).
2. Testing the agent with a budget of 2500 test cases, with both Random Testing and MDPFuzz. The results of Random Testing are recorded in the current directory prefixed by `random_testing`, while the results of MDPFuzz are prefixed by `mdpfuzz`.
3. Plotting the evolution of the number of failures found over the test cases by the two methods in the file `failure_results_comparison_demo.png`.