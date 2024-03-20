# Replicate of MDPFuzz

This repository aims at re-implementing the framework [MDPFuzz](https://sites.google.com/view/mdpfuzz/evaluation-results?authuser=0), whose original code can be found [here](https://github.com/Qi-Pang/MDPFuzz).
This tool has been used in the paper *Testing MDP Solving Models with MDPFuzz: A Replicability Study*.

## Content

The implementation can be found under ``src/``.
Precisely, the *Fuzzer* class (in ``src/mdpfuzz.py``) provides functions for fuzzing with and without GMM guidance, as well as a simple random testing procedure.

## Installation

Install the package with:
```
pip install git+https://github.com/QuentinMaz/Replicability_Study
```
Alternatively, you can use locally the code by cloning the repository (with``git clone https://github.com/QuentinMaz/Replicability_Study``) and import the classes by appending the folder ``src/`` folder to your path.