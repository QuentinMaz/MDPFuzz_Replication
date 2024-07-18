# Replicate of MDPFuzz
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

This repository contains a re-implementation the policy testing framework [MDPFuzz](https://sites.google.com/view/mdpfuzz/evaluation-results?authuser=0), whose original code can be found [here](https://github.com/Qi-Pang/MDPFuzz).
This tool has been used in the paper *Replicability Study: Policy Testing with MDPFuzz*.

## Installation

You can install the package with `pip`:
```
pip install git+https://github.com/QuentinMaz/MDPFuzz_Replication
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

#### Setup

Setup a virtual environment with, for instance, (conda)[https://docs.conda.io/en/latest/]:
```
conda create -n demo python=3.10.12
conda activate demo
pip install git+https://github.com/QuentinMaz/MDPFuzz_Replication
pip install gymnasium==0.29.1
pip install stable-baselines3==2.2.1
```