import os
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

import time
import copy
import tqdm
import numpy as np

from scipy.stats import multivariate_normal
from typing import List, Tuple
from gmm import CoverageModel


class Pool():
    def __init__(self) -> None:
        self.inputs = []
        self.rewards = []
        self.oracles = []
        self.sensitivities = []
        self.coverages = []

    def add(self, input: np.ndarray, acc_reward: float, coverage: float, sensitivity: float):
        pass

    def select(self):
        '''Returns one of the inputs along with its accumulated reward of the pool with random sampling biased by the sensitivities.'''
        pass


class Fuzzer():
    def __init__(self, random_seed: int, k: int, tau: float, gamma: float) -> None:
        # in order, k is the number of components (or clusters) for the 2 GMMs, tau is the density threshold to update the GMMs and gamma is the weight of the update
        self.k = k
        self.tau = tau
        self.gamma = gamma

        # random generators
        self.random_seed = random_seed
        self.rng: np.random.Generator = np.random.default_rng(self.random_seed)

        # coverage model (composed of 2 GMMS)
        self.coverage_model = CoverageModel(random_seed, k, gamma)


    def _concatenate_state_sequence(self, state_sequence: np.ndarray) -> np.ndarray:
        data_concat = []
        for i in range(len(state_sequence) - 1):
            data_concat.append(np.hstack([state_sequence[i], state_sequence[i+1]]))
        return np.array(data_concat)


    def sampling(self, n: int = 1) -> List[np.ndarray]:
        '''Returns a list of @n inputs randomly generated.'''
        return NotImplementedError('Use-case dependent.')


    def mutate_validate(self):
        return NotImplementedError('Use-case dependent.')


    def mdp(self, input, policy) -> Tuple[float, bool, List[np.ndarray]]:
        return NotImplementedError('Use-case dependent.')


    def sentivity(self, state: np.ndarray, acc_reward: float = None, policy = None) -> Tuple[float, float, List[np.ndarray]]:
        '''
        Computes the sensitivity of the state @state.
        It first perturbs the state and computes the perturbation quantity.
        Then, the two states are executed and the sensitivity is computed.
        It returns the latter, as well as the accumulated reward for the state and the state sequence.
        '''
        # perturbs the state and computes the perturbation
        perturbed_state = self.mutate_validate(state)
        perturbation = np.linalg.norm(state - perturbed_state)

        # runs the two states if no accumulated reward is provided
        if acc_reward is None:
            acc_reward, crash, state_sequence = self.mdp(state, policy)
        else:
            state_sequence = []
        acc_reward_perturbed, _crash_perturbed, _state_sequence_perturbed = self.mdp(state, policy)

        # computes the sensitivity, the coverage, and adds test case in the pool
        sensitivity = np.abs(acc_reward - acc_reward_perturbed) / perturbation

        return sensitivity, acc_reward, state_sequence


    def record_crash(self):
        pass


    def initialize_coverage_model(self, **kwargs):
        '''Initializes the coverage model.'''

        state_sequence = kwargs.get('state_sequence', None)
        if state_sequence is None:
            policy = kwargs.get('policy', None)
            assert policy is not None
            random_input = kwargs.get('input', self.sampling())
            _acc_reward, _crash, state_sequence = self.mdp(random_input, policy)

        self.coverage_model.initialize(state_sequence)
        print('Coverage model initialized')


    def fuzzing(self, n: int, test_budget: int, policy):
        initial_inputs = self.sampling(n)
        pool = Pool()
        # initialization
        self.initialize_coverage_model(policy=policy)
        pbar = tqdm.tqdm(total=n)
        for state in initial_inputs:
            sensitivity, acc_reward, state_sequence = self.sentivity(state)
            state_sequence_conc = self._concatenate_state_sequence(state_sequence)
            # computes the coverage and adds test case in the pool
            coverage = self.coverage_model.sequence_freshness(state_sequence, state_sequence_conc, tau=self.tau)
            pool.add(state, acc_reward, coverage, sensitivity)
            pbar.update(1)
        pbar.close()

        start_time = time.time()
        current_time = time.time()
        test_budget_in_seconds = 3600 * test_budget
        pbar = tqdm.tqdm(total=test_budget_in_seconds)

        while (current_time - start_time) > test_budget_in_seconds:
            input, acc_reward_input = pool.select()
            mutant = self.mutate_validate(input)
            acc_reward_mutant, crash, state_sequence = self.mdp(mutant, policy)
            state_sequence_conc = self._concatenate_state_sequence(state_sequence)
            coverage = self.coverage_model.sequence_freshness(state_sequence, state_sequence_conc, tau=self.tau)
            if crash:
                self.record_crash(mutant)
            elif (acc_reward_mutant < acc_reward_input) or (coverage < self.tau):
                sensitivity, _acc_reward_mutant_copy, _empty_list = self.sentivity(mutant, acc_reward_mutant)
                pool.add(mutant, acc_reward_mutant, coverage, sensitivity)

            pbar.update(int(time.time() - current_time))
            current_time = time.time()

        pbar.close()


if __name__ == '__main__':
    print('To test.')