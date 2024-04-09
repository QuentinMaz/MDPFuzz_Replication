# Replicate of MDPFuzz

This repository aims at re-implementing the framework [MDPFuzz](https://sites.google.com/view/mdpfuzz/evaluation-results?authuser=0), whose original code can be found [here](https://github.com/Qi-Pang/MDPFuzz).
This tool has been used in the paper *Replicability Study: Policy Testing with MDPFuzz*.

## Content

The implementation can be found under `src/`.
Precisely, the *Fuzzer* class (in `src/mdpfuzz.py`) provides functions for fuzzing with and without GMM guidance, as well as a simple random testing procedure.

## Installation

You can either install the package with `pip` or use the code locally by cloning the repository and importing the classes.
In the latter case, don't forget to append the folder `src/` to your path.