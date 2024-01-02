import os
os.environ["MKL_NUM_THREADS"] = '1'
os.environ["NUMEXPR_NUM_THREADS"] = '1'
os.environ["OMP_NUM_THREADS"] = '1'

import time
import copy
import tqdm
import numpy as np

from scipy.stats import multivariate_normal
from typing import List, Tuple


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
        self.normal = multivariate_normal
        self.normal.random_state = np.random.default_rng(self.random_seed)


    def _generate_empty_parameters(self):
        return {i: [None for _ in range(self.k)] for i in range(3)}


    def sampling(self, n: int) -> List[np.ndarray]:
        '''Returns a list of @n inputs randomly generated.'''
        return NotImplementedError('Use-case dependent.')


    def mutate_validate(self):
        return NotImplementedError('Use-case dependent.')


    def sequence_freshness(self, state_sequence: List[np.ndarray], params_s: dict, params_c: dict):
        first_state = state_sequence[0]
        prob = self.gmm(first_state, params_s)

        # indices are shifted compared to the definition
        for i in range(1, len(state_sequence)):
            prob *= (self.gmm(state_sequence[i], params_s) / self.gmm(np.hstack([state_sequence[i-1], state_sequence[i]]), params_c))

        assert prob > 0.0 and prob < 1.0

        if prob < self.tau:
            updated_params_s, updated_params_c = self.dynamic_EM(state_sequence, params_s, params_c)
            return prob, updated_params_s, updated_params_c
        else:
            prob, params_s, params_c


    def mdp(self) -> Tuple[float, bool, List[np.ndarray]]:
        return NotImplementedError('Use-case dependent.')


    def sentivity(self, state: np.ndarray, acc_reward: float = None) -> Tuple[float, float, List[np.ndarray]]:
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
            acc_reward, crash, state_sequence = self.mdp(state)
        else:
            state_sequence = []
        acc_reward_perturbed, _crash_perturbed, _state_sequence_perturbed = self.mdp(state)

        # computes the sensitivity, the coverage, and adds test case in the pool
        sensitivity = np.abs(acc_reward - acc_reward_perturbed) / perturbation

        return sensitivity, acc_reward, state_sequence


    def record_crash(self):
        pass

    def fuzzing(self, n: int, test_budget: int):
        initial_inputs = self.sampling(n)
        params_s, params_c = self.init()
        pool = Pool()

        pbar = tqdm.tqdm(total=n)
        for state in initial_inputs:
            sensitivity, acc_reward, state_sequence = self.sentivity(state)
            # computes the coverage and adds test case in the pool
            coverage, params_s, params_c = self.sequence_freshness(state_sequence, params_s, params_c)
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
            acc_reward_mutant, crash, state_sequence = self.mdp(mutant)
            coverage, params_s, params_c = self.sequence_freshness(state_sequence, params_s, params_c)
            if crash:
                self.record_crash(mutant)
            elif (acc_reward_mutant < acc_reward_input) or (coverage < self.tau):
                sensitivity, _acc_reward_mutant_copy, _empty_list = self.sentivity(mutant, acc_reward_mutant)
                pool.add(mutant, acc_reward_mutant, coverage, sensitivity)

            pbar.update(int(time.time() - current_time))
            current_time = time.time()

        pbar.close()


    def fuzzing(self, n: int, test_budget: int):
        '''Optimized version which does not strictly mimic the algorithm.'''
        initial_inputs = self.sampling(n)
        params_s, params_c = self.init()
        pool = Pool()

        pbar = tqdm.tqdm(total=n)
        for state in initial_inputs:
            sensitivity, acc_reward, state_sequence = self.sentivity(state)
            # computes the coverage and adds test case in the pool
            coverage, params_s, params_c = self.sequence_freshness(state_sequence, params_s, params_c)
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
            acc_reward_mutant, crash, state_sequence = self.mdp(mutant)
            coverage, params_s, params_c = self.sequence_freshness(state_sequence, params_s, params_c)
            if crash:
                self.record_crash(mutant)
            elif (acc_reward_mutant < acc_reward_input) or (coverage < self.tau):
                sensitivity, _acc_reward_mutant_copy, _empty_list = self.sentivity(mutant, acc_reward_mutant)
                pool.add(mutant, acc_reward_mutant, coverage, sensitivity)

            pbar.update(int(time.time() - current_time))
            current_time = time.time()

        pbar.close()

    #################### GMM AND DYN EM MANAGEMENT ######################


    def init(self):
        '''Initializes the S-C parameters and returns the latter as dictionaries.'''
        states = self.sampling(self.k + 1)
        params_s = self._generate_empty_parameters()
        params_c = self._generate_empty_parameters()
        #TODO: is there any possible reference-related issues in the computation?
        for i in range(self.k):
            for state, params in zip([states[i], np.hstack([states[i], states[i+1]])], [params_s, params_c]):
                params[0][i] = 1 / self.k
                params[1][i] = state
                params[2][i] = np.matmul(state, state.T)
        return params_s, params_c


    def get_gmm_parameters(self, params: dict):
        pi_list = []
        mu_list = []
        sigma_list = []
        for i in range(self.k):
            pi = params[0][i]
            mu = params[1][i] / params[0][i]
            sigma = (params[2][i] - np.matmul(mu, params[1][i].T)) / params[0][i]

            pi_list.append(pi)
            mu_list.append(mu)
            sigma_list.append(sigma)
        return pi_list, mu_list, sigma_list


    def gmm(self, state: np.ndarray, params: dict):
        pi_list, mu_list, sigma_list = self.get_gmm_parameters(params)
        total = 0.0
        for i in range(self.k):
            total += pi_list[i] * self.normal.pdf(state, mean=mu_list[i], cov=sigma_list[i])
        return total


    def get_cs_statistics(self, state: np.ndarray, params: dict):
        #TODO: following the appendix leads to unfolding dictionaries to list and vice-versa...
        pi_list, mu_list, sigma_list = self.get_gmm_parameters(params)
        statistics = self._generate_empty_parameters()
        for i in range(self.k):
            weight = pi_list[i] * self.normal.pdf(state, mean=mu_list[i], cov=sigma_list[i])
            statistics[0][i] = weight
            statistics[1][i] = weight * state
            statistics[2][i] = weight * np.matmul(state, state.T)
        return statistics


    def update_parameters(self, cs_statistics: dict, params: dict):
        updated_params = self._generate_empty_parameters()
        for i in range(self.k):
            updated_params[0][i] = self.gamma * cs_statistics[0][i] + (1 - self.gamma) * params[0][i]
            updated_params[1][i] = self.gamma * cs_statistics[1][i] + (1 - self.gamma) * params[1][i]
            updated_params[2][i] = self.gamma * cs_statistics[0][i] + (1 - self.gamma) * params[0][i]
        return updated_params


    def dynamic_EM(self, state_sequence: List[np.ndarray], params_s: dict, params_c: dict):
        '''
        Updates the parameters with the state sequence.
        The update is dynamic and online throughout the states.
        Note that the parameters are copied before the update, and correspond to the ones returned.
        '''
        # avoids changing the input dictionaries
        updated_params_s = copy.deepcopy(params_s)
        updated_params_c = copy.deepcopy(params_c)
        # as defined the paper, the parameters are being updated throughout the loop (state sequence processing)
        for i in range(len(state_sequence) - 1):
            state_s = state_sequence[i]
            cs_statistics_s = self.get_cs_statistics(state_s, updated_params_s)
            self.update_parameters(cs_statistics_s, updated_params_s)

            state_c = np.hstack([state_sequence[i], state_sequence[i+1]])
            cs_statistics_c = self.get_cs_statistics(state_c, updated_params_c)
            self.update_parameters(cs_statistics_c, updated_params_c)
        return updated_params_s, updated_params_c


if __name__ == '__main__':
    print('To test.')