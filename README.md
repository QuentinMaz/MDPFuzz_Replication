# Replicate of MDPFuzz
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

This repository contains a re-implementation the policy testing framework [MDPFuzz](https://sites.google.com/view/mdpfuzz/evaluation-results?authuser=0), whose original code can be found [here](https://github.com/Qi-Pang/MDPFuzz).
This tool has been used in the paper *Replicability Study: Policy Testing with MDPFuzz*.

## Content

The implementation can be found under `mdpfuzz/`.
Precisely, the *Fuzzer* class (`mdpfuzz/mdpfuzz.py`) provides functions for fuzzing with and without GMM guidance, as well as a simple random testing procedure.

## Installation

You can either install the package with `pip`:
```
pip install git+https://github.com/QuentinMaz/MDPFuzz_Replicate
```
Or use the code locally by cloning the repository and importing the classes.
In the latter case, don't forget to append the folder `mdpfuzz/` to your path.