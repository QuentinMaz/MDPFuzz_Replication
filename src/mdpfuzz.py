import os
import time
import json
import tqdm
import numpy as np

from typing import List, Tuple, Dict, Any

# or runs python -m mdpfuzz.py
if __package__ is None or __package__ == '':
    # uses current directory visibility
    from gmm import CoverageModel
    from logger import FuzzerLogger
    from executor import Executor
else:
    from .gmm import CoverageModel
    from .logger import FuzzerLogger
    from .executor import Executor

class Pool():
    '''Compared to the paper, the pool features the storage of crashes, which correspond to failure-triggering states/inputs.'''
    def __init__(self, is_integer: bool = False) -> None:
        self.inputs: List[np.ndarray] = []
        self.rewards: List[float] = []
        self.oracles: List[int] = []
        self.sensitivities: List[float] = []
        self.coverages: List[float] = []
        self.indices: List[str] = []
        self.added: List[int] = [] # tracks insertion in the pool
        self.selected: List[int] = [] # tracks selection in the pool
        self.crashes: List[np.ndarray] = []

        self.delimiter = ' '
        self.is_integer = is_integer


    def _key(self, input: np.ndarray) -> str:
        '''Returns the "key" format of an input.'''
        if self.is_integer:
            return self.delimiter.join([f'{i:.0f}' for i in input])
        else:
            return self.delimiter.join([str(i) for i in input])


    def add(self, input: np.ndarray, acc_reward: float, coverage: float, sensitivity: float, oracle: bool):
        '''
        Adds a test case result to the pool.
        If the input/state has already been evaluated, the results are erased.
        '''
        key = self._key(input)
        try:
            index = self.indices.index(key)
        except ValueError:
            index = None

        if index is None:
            self.indices.append(key)
            self.inputs.append(input)
            self.rewards.append(acc_reward)
            self.coverages.append(coverage)
            self.sensitivities.append(sensitivity)
            self.added.append(0)
            self.selected.append(0)
            self.oracles.append(int(oracle))
        else:
            # print(f'input {key} is already in the pool!')
            self.rewards[index] = acc_reward
            self.coverages[index] = coverage
            self.sensitivities[index] = sensitivity
            self.added[index] += 1
            self.oracles[index] = int(oracle)
            # does nothing for selection tracking


    def add_crash(self, state: np.ndarray):
        '''Keeps track of failure-triggering states by adding @state to the list of crashes.'''
        self.crashes.append(state.copy())


    def select(self, rng: np.random.Generator) -> Tuple[np.ndarray, float]:
        '''Returns one of the inputs along with its accumulated reward of the pool with random sampling biased by the sensitivities.'''
        # the sum of sensitivities can be zero?
        index = rng.choice(len(self.inputs), p=(self.sensitivities / np.sum(self.sensitivities)))
        self.selected[index] += 1
        # copy.deepcopy(self.inputs[index]) # .copy(() seems to be enough
        return self.inputs[index].copy(), self.rewards[index]


    def save(self, path: str):
        '''Saves all the lists of the pool instance.'''
        # if Python 3.8 and higher
        # x = 5
        # variable_name = f"{x=}".split("=")[0]
        # print(variable_name)
        if self.is_integer:
            np.savetxt(path + '_inputs.txt', self.inputs, fmt='%1.0f', delimiter=',')
            np.savetxt(path + '_crashes.txt', self.crashes, fmt='%1.0f', delimiter=',')
        else:
            np.savetxt(path + '_inputs.txt', self.inputs, delimiter=',')
            np.savetxt(path + '_crashes.txt', self.crashes, delimiter=',')

        np.savetxt(path + '_added.txt', self.added, fmt='%1.0f', delimiter=',')
        np.savetxt(path + '_selected.txt', self.selected, fmt='%1.0f', delimiter=',')
        np.savetxt(path + '_oracles.txt', self.oracles, fmt='%1.0f', delimiter=',')
        np.savetxt(path + '_rewards.txt', self.rewards, delimiter=',')
        np.savetxt(path + '_sensitivities.txt', self.sensitivities, delimiter=',')
        np.savetxt(path + '_coverages.txt', self.coverages, delimiter=',')


    #TODO: loading integer values does not work (but it is not an used feature)
    def load(self, path: str, as_integers: bool = False):
        '''Loads the results of a pool instance by loading all the lists.'''
        self.is_integer = as_integers
        if self.is_integer:
            self.is_integer = True
            self.inputs = [i for i in np.loadtxt(path + '_inputs.txt', delimiter=',', dtype=int)]
            self.crashes = [i for i in np.loadtxt(path + '_crashes.txt', delimiter=',', dtype=int)]
        else:
            self.inputs = [i for i in np.loadtxt(path + '_inputs.txt', delimiter=',', dtype=float)]
            self.crashes = [i for i in np.loadtxt(path + '_crashes.txt', delimiter=',', dtype=float)]

        self.indices = [self._key(i) for i in self.inputs]
        self.added = [i for i in np.loadtxt(path + '_added.txt', delimiter=',', dtype=int)]
        self.selected = [i for i in np.loadtxt(path + '_selected.txt', delimiter=',', dtype=int)]
        self.oracles = [i for i in np.loadtxt(path + '_oracles.txt', delimiter=',', dtype=int)]
        self.rewards = [i for i in np.loadtxt(path + '_rewards.txt', delimiter=',', dtype=float)]
        self.sensitivities = [i for i in np.loadtxt(path + '_sensitivities.txt', delimiter=',', dtype=float)]
        self.coverages = [i for i in np.loadtxt(path + '_coverages.txt', delimiter=',', dtype=float)]

        tmp = len(self.inputs)
        assert len(self.added) == tmp
        assert len(self.selected) == tmp
        assert len(self.indices) == tmp
        assert len(self.rewards) == tmp
        assert len(self.sensitivities) == tmp
        assert len(self.coverages) == tmp
        assert len(self.oracles) == tmp


class Fuzzer():
    def __init__(self, random_seed: int, k: int, tau: float, gamma: float, executor: Executor) -> None:
        # in order, k is the number of components (or clusters) for the 2 GMMs, tau is the density threshold to update the GMMs and gamma is the weight of the update
        self.k = k
        self.tau = tau
        self.gamma = gamma

        # random generators
        self.random_seed = random_seed
        self.rng: np.random.Generator = np.random.default_rng(self.random_seed)

        # coverage model (composed of 2 GMMS)
        self.coverage_model = CoverageModel(random_seed, k, gamma)
        # used to track uniqueness of solutions
        self.evaluated_solutions = []

        self.executor = executor
        self.sim_steps = self.executor.sim_steps
        self.env_seed = self.executor.env_seed


    def _concatenate_state_sequence(self, state_sequence: np.ndarray) -> np.ndarray:
        data_concat = []
        for i in range(len(state_sequence) - 1):
            data_concat.append(np.hstack([state_sequence[i], state_sequence[i+1]]))
        return np.array(data_concat)


    def sampling(self, n: int = 1) -> List[np.ndarray]:
        '''Returns a list of @n inputs randomly generated.'''
        if n == 1:
            return self.executor.generate_input(self.rng)
        else:
            return self.executor.generate_inputs(self.rng, n=n)


    def mutate(self, state: np.ndarray):
        return self.executor.mutate(state, self.rng)


    def mutate_validate(self, state: np.ndarray):
        attempts = 1
        while attempts < 100:
            mutate_states = self.mutate(state)
            tmp = mutate_states.tolist()
            if not (tmp in self.evaluated_solutions):
                self.evaluated_solutions.append(tmp)
                break
            else:
                attempts += 1
        return mutate_states


    def mdp(self, state: np.ndarray, policy: Any = None) -> Tuple[float, bool, np.ndarray]:
        '''Returns the accumulated reward, whether a crash is detected and the state sequence.'''
        episode_reward, done, obs_seq, _exec_time = self.executor.execute_policy(state, policy)
        return episode_reward, done, obs_seq


    def sentivity(self, state: np.ndarray, acc_reward: float = None, policy: Any = None) -> Tuple[float, float, bool, List[np.ndarray]]:
        '''
        Computes the sensitivity of the state @state.
        It first perturbs the state and computes the perturbation quantity.
        Then, the two states are executed and the sensitivity is computed.
        It returns the latter, as well as the results of the execution for the state (acc. reward, sequence and oracle).
        '''
        # perturbs the state and computes the perturbation
        perturbed_state = self.mutate_validate(state)
        perturbation = np.linalg.norm(state - perturbed_state)

        # runs the two states if no accumulated reward is provided
        if acc_reward is None:
            acc_reward, crash, state_sequence = self.mdp(state, policy)
        else:
            state_sequence = []
            crash = None
        acc_reward_perturbed, crash_perturbed, _state_sequence_perturbed = self.mdp(perturbed_state, policy)

        if self.logger is not None:
            self.logger.log(input=perturbed_state, oracle=crash_perturbed, reward=acc_reward_perturbed)

        # computes the sensitivity, the coverage, and adds test case in the pool
        sensitivity = np.abs(acc_reward - acc_reward_perturbed) / perturbation

        return sensitivity, acc_reward, crash, state_sequence


    def local_sensitivity(self, state: np.ndarray, state_mutate: np.ndarray, state_reward: float, state_mutate_reward: float):
        perturbation = np.linalg.norm(state - state_mutate)
        return np.abs(state_reward - state_mutate_reward) / perturbation


    def initialize_coverage_model(self, **kwargs):
        '''Initializes the coverage model.'''

        state_sequence = kwargs.pop('state_sequence', None)
        if state_sequence is None:
            policy = kwargs.get('policy', None)
            random_input = kwargs.get('input', self.sampling())
            reward, crash, state_sequence = self.mdp(random_input, policy)

            if self.logger is not None:
                self.logger.log(input=random_input, oracle=crash, reward=reward)

        # it needs at least k + 1 states (for gmm_c)
        if len(state_sequence) < self.k + 1:
            return self.initialize_coverage_model(**kwargs)
        else:
            self.coverage_model.initialize(state_sequence)
        print('Coverage model initialized')


    def fuzzing(self, n: int, policy: Any = None, **kwargs):
        '''
        Conducts fuzzing to generate test cases for the system under test (SUT).

        Args:
        - n (int): Number of iterations for fuzzing.
        - policy (tt.TestAgent): The testing policy or agent guiding the fuzzing process.
        - saving_path (str, optional): Path to save logs and results (default: None).
        - local_sensitivity (bool, optional): Flag indicating whether to compute local sensitivity (default: False).
        - test_budget_in_seconds (int, optional): Time budget for fuzzing in seconds (default: None).
        - test_budget (int, optional): Number of iterations if time budget is not specified (default: None).

        Returns:
        None. The function conducts the fuzzing process and stores generated test cases.
        '''
        path = kwargs.get('saving_path', None)
        if path is not None:
            self.logger = FuzzerLogger(path + '_logs.txt')
            self.logger.log_header_line()
        else:
            self.logger = None

        local_sensitivity = kwargs.get('local_sensitivity', False)

        initial_inputs = self.sampling(n)
        pool = Pool(is_integer=np.issubdtype(initial_inputs.dtype, np.integer))
        # initializes the coverage model by running the policy on a randomly generated input to sample states of the MDP
        self.initialize_coverage_model(policy=policy)
        pbar = tqdm.tqdm(total=n)
        for state in initial_inputs:
            sensitivity, acc_reward, oracle, state_sequence = self.sentivity(state, policy=policy)
            state_sequence_conc = self._concatenate_state_sequence(state_sequence)
            # computes the coverage and adds test case in the pool
            coverage = self.coverage_model.sequence_freshness(state_sequence, state_sequence_conc, tau=self.tau)
            pool.add(state, acc_reward, coverage, sensitivity, oracle)

            if self.logger is not None:
                self.logger.log(input=state, oracle=oracle, reward=acc_reward, sensitivity=sensitivity, coverage=coverage)

            if oracle:
                pool.add_crash(state)

            pbar.update(1)
        pbar.close()

        test_budget_in_seconds = kwargs.get('test_budget_in_seconds', None)
        if test_budget_in_seconds is None:
            test_budget = kwargs.get('test_budget', None)
            assert test_budget is not None
            # accounts for the cost of the initialization
            test_budget -=  (2 * n) + 1
            pbar = tqdm.tqdm(total=test_budget)
            num_iterations = 0
        else:
            start_time = time.time()
            current_time = time.time()
            seconds = 0
            pbar = tqdm.tqdm(total=test_budget_in_seconds)

        try:
            while True:
                if test_budget_in_seconds is None:
                    if num_iterations == test_budget:
                        break
                else:
                    if (current_time - start_time) > test_budget_in_seconds:
                        break

                input, acc_reward_input = pool.select(self.rng)
                mutant = self.mutate_validate(input)
                acc_reward_mutant, oracle, state_sequence = self.mdp(mutant, policy)
                state_sequence_conc = self._concatenate_state_sequence(state_sequence)
                coverage = self.coverage_model.sequence_freshness(state_sequence, state_sequence_conc, tau=self.tau)
                sensitivity = None
                if oracle:
                    pool.add_crash(mutant)
                elif (acc_reward_mutant < acc_reward_input) or (coverage < self.tau):
                    if local_sensitivity:
                        sensitivity = self.local_sensitivity(input, mutant, acc_reward_input, acc_reward_mutant)
                    else:
                        sensitivity, _acc_reward_mutant_copy, _none, _empty_list = self.sentivity(mutant, acc_reward=acc_reward_mutant, policy=policy)
                    pool.add(mutant, acc_reward_mutant, coverage, sensitivity, oracle)

                if self.logger is not None:
                    self.logger.log(input=mutant, oracle=oracle, reward=acc_reward_mutant, sensitivity=sensitivity, coverage=coverage)

                if test_budget_in_seconds is None:
                    num_iterations += 1
                    pbar.update(1)
                else:
                    current_time = time.time()
                    if int(current_time - start_time) > seconds:
                        seconds += 1
                        pbar.update(1)
        except Exception as e:
            print(e)

        pbar.close()
        if path is not None:
            pool.save(path)
            self.save_configuration(path)
            self.save_evaluated_solutions(path)
            self.coverage_model.save(path)


    def fuzzing_no_coverage(self, n: int, policy: Any = None, **kwargs):
        '''
        Works similarly as fuzzing but coverages are not computed.
        '''
        path = kwargs.get('saving_path', None)
        if path is not None:
            self.logger = FuzzerLogger(path + '_logs.txt')
            self.logger.log_header_line()
        else:
            self.logger = None

        local_sensitivity = kwargs.get('local_sensitivity', False)

        initial_inputs = self.sampling(n)
        pool = Pool(is_integer=np.issubdtype(initial_inputs.dtype, np.integer))
        pbar = tqdm.tqdm(total=n)
        for state in initial_inputs:
            sensitivity, acc_reward, oracle, state_sequence = self.sentivity(state, policy=policy)
            pool.add(state, acc_reward, 0, sensitivity, oracle)

            if self.logger is not None:
                self.logger.log(input=state, oracle=oracle, reward=acc_reward, sensitivity=sensitivity)

            if oracle:
                pool.add_crash(state)

            pbar.update(1)
        pbar.close()

        test_budget_in_seconds = kwargs.get('test_budget_in_seconds', None)
        if test_budget_in_seconds is None:
            test_budget = kwargs.get('test_budget', None)
            assert test_budget is not None
            # accounts for the cost of the initialization
            test_budget -=  (2 * n) + 1
            pbar = tqdm.tqdm(total=test_budget)
            num_iterations = 0
        else:
            start_time = time.time()
            current_time = time.time()
            seconds = 0
            pbar = tqdm.tqdm(total=test_budget_in_seconds)

        while True:
            if test_budget_in_seconds is None:
                if num_iterations == test_budget:
                    break
            else:
                if (current_time - start_time) > test_budget_in_seconds:
                    break

            input, acc_reward_input = pool.select(self.rng)
            mutant = self.mutate_validate(input)
            acc_reward_mutant, oracle, state_sequence = self.mdp(mutant, policy)
            sensitivity = None
            if oracle:
                pool.add_crash(mutant)
            elif acc_reward_mutant < acc_reward_input:
                if local_sensitivity:
                    sensitivity = self.local_sensitivity(input, mutant, acc_reward_input, acc_reward_mutant)
                else:
                    sensitivity, _acc_reward_mutant_copy, _none, _empty_list = self.sentivity(mutant, acc_reward=acc_reward_mutant, policy=policy)
                pool.add(mutant, acc_reward_mutant, 0, sensitivity, oracle)

            if self.logger is not None:
                self.logger.log(input=mutant, oracle=oracle, reward=acc_reward_mutant, sensitivity=sensitivity)

            if test_budget_in_seconds is None:
                num_iterations += 1
                pbar.update(1)
            else:
                current_time = time.time()
                if int(current_time - start_time) > seconds:
                    seconds += 1
                    pbar.update(1)

        pbar.close()
        if path is not None:
            pool.save(path)
            self.save_configuration(path)
            self.save_evaluated_solutions(path)


    def resume_fuzzing(self, loading_path: str, policy: Any = None, **kwargs):
        '''Not tested.'''
        pool = Pool()
        pool.load(loading_path)
        self.coverage_model.load(loading_path)
        self.load(loading_path)

        path = kwargs.get('saving_path', None)
        if path is not None:
            self.logger = FuzzerLogger(path + '_logs.txt')
            self.logger.log_header_line()
        else:
            self.logger = None

        local_sensitivity = kwargs.get('local_sensitivity', False)

        test_budget_in_seconds = kwargs.get('test_budget_in_seconds', None)
        if test_budget_in_seconds is None:
            test_budget = kwargs.get('test_budget', None)
            assert test_budget is not None
            pbar = tqdm.tqdm(total=test_budget)
            num_iterations = 0
        else:
            start_time = time.time()
            current_time = time.time()
            seconds = 0
            pbar = tqdm.tqdm(total=test_budget_in_seconds)

        while True:
            if test_budget_in_seconds is None:
                if num_iterations == test_budget:
                    break
            else:
                if (current_time - start_time) > test_budget_in_seconds:
                    break

            input, acc_reward_input = pool.select(self.rng)
            mutant = self.mutate_validate(input)
            acc_reward_mutant, oracle, state_sequence = self.mdp(mutant, policy)
            state_sequence_conc = self._concatenate_state_sequence(state_sequence)
            coverage = self.coverage_model.sequence_freshness(state_sequence, state_sequence_conc, tau=self.tau)
            sensitivity = None
            if oracle:
                pool.add_crash(mutant)
            elif (acc_reward_mutant < acc_reward_input) or (coverage < self.tau):
                if local_sensitivity:
                    sensitivity = self.local_sensitivity(input, mutant, acc_reward_input, acc_reward_mutant)
                else:
                    sensitivity, _acc_reward_mutant_copy, _none, _empty_list = self.sentivity(mutant, acc_reward=acc_reward_mutant, policy=policy)
                pool.add(mutant, acc_reward_mutant, coverage, sensitivity, oracle)

            if self.logger is not None:
                self.logger.log(input=mutant, oracle=oracle, reward=acc_reward_mutant, sensitivity=sensitivity, coverage=coverage)

            if test_budget_in_seconds is None:
                num_iterations += 1
                pbar.update(1)
            else:
                current_time = time.time()
                if int(current_time - start_time) > seconds:
                    seconds += 1
                    pbar.update(1)

        pbar.close()
        if path is not None:
            pool.save(path)
            self.save_configuration(path)
            self.save_evaluated_solutions(path)
            self.coverage_model.save(path)


    def save_configuration(self, path: str):
        filepath = path.split('.json')[0]
        configuration = dict()
        # attributes
        configuration['k'] = self.k
        configuration['gamma'] = self.gamma
        configuration['tau'] = self.tau
        configuration['random_seed'] = self.random_seed
        configuration['env_seed'] = self.env_seed
        # along with the initial random seed and the states of the random generator
        configuration['random_state'] = self.rng.bit_generator.state
        with open(filepath + '_config.json', 'w') as f:
            f.write(json.dumps(configuration))


    def save_evaluated_solutions(self, path: str):
        evaluations = np.array(self.evaluated_solutions)
        if np.issubdtype(evaluations.dtype, np.integer):
            np.savetxt(path + '_evaluations.txt', evaluations, fmt='%1.0f', delimiter=',')
        else:
            np.savetxt(path + '_evaluations.txt', evaluations, delimiter=',')


    def load(self, path: str):
        self.coverage_model.load(path)
        config_filepath = path + '_config.json'
        assert os.path.isfile(config_filepath), config_filepath
        with open(config_filepath, 'r') as f:
            config = json.load(f)
        self._load_dict(config)
        if os.path.isfile(path + '_evaluations.txt'):
            self.load_evaluated_solutions(path + '_evaluations.txt')
            print(f'found {len(self.evaluated_solutions)} evaluated solutions.')


    def _load_dict(self, configuration: Dict):
        self.k = configuration['k']
        self.gamma = configuration['gamma']
        self.random_seed = configuration['random_seed']
        self.env_seed = configuration['env_seed']
        self.rng: np.random.Generator = np.random.default_rng(self.random_seed)
        self.rng.bit_generator.state = configuration['random_state']


    def load_evaluated_solutions(self, filepath: str):
        self.evaluated_solutions = np.loadtxt(filepath, delimiter=',', dtype=int).tolist()


    def random_testing(self, n: int, policy: Any = None, path: str = 'logs'):
        '''RT baseline that generates an input at each iteration until it has not been tested yet.'''
        self.logger = FuzzerLogger(path + '_logs.txt')
        self.logger.log_header_line()

        pbar = tqdm.tqdm(total=n)
        i = 0
        while i < n:
            random_input = self.sampling(1)
            tmp = random_input.tolist()
            if not (tmp in self.evaluated_solutions):
                self.evaluated_solutions.append(tmp)
                acc_reward, oracle, _state_sequence = self.mdp(random_input, policy)
                self.logger.log(input=random_input, oracle=oracle, reward=acc_reward)
                pbar.update(1)
                i += 1
        pbar.close()

        self.save_configuration(path)
        self.save_evaluated_solutions(path)