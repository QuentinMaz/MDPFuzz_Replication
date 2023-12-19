import os
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

import time
import copy
import tqdm
import json
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import multivariate_normal
from typing import List, Tuple, Dict
from utils import modify_covariance_matrix, plot_gaussians, generate_clustered_data, create_gif, remove_gaussian


class GMM():
    '''CS statistics are weighted.'''
    def __init__(self, random_seed: int, k: int):
        self.k = k
        self.dim: int = None
        self.random_seed = random_seed
        self.rng: np.random.Generator = np.random.default_rng(self.random_seed)
        self.normal = multivariate_normal
        self.normal.random_state = np.random.default_rng(self.random_seed)

        self.coefficients: np.ndarray = None
        self.means: np.ndarray = None
        self.covariances: np.ndarray = None
        self.cs_means: np.ndarray = None
        self.cs_squares: np.ndarray = None


    def initialize(self, data: np.ndarray):
        '''
        Sets the different attributes of the class.
        Means are weighted '''
        assert len(data) == self.k
        self.dim: int = data.shape[1]
        self.coefficients: np.ndarray = np.ones(self.k) / self.k
        self.means: np.ndarray = copy.deepcopy(data)

        self.covariances: np.ndarray = np.zeros((self.k, self.dim, self.dim))
        self.cs_means: np.ndarray = np.zeros((self.k, self.dim))
        self.cs_squares: np.ndarray = np.zeros((self.k, self.dim, self.dim))
        for i in range(self.k):
            self.covariances[i] = np.eye(self.dim)
            self.cs_squares[i] = np.matmul(data[i:i+1].T, data[i:i+1]) * self.coefficients[0]
            self.cs_means[i] = self.means[i].copy() * self.coefficients[0]


    def update_cs_statistics(self, point: np.ndarray, gamma: float):
        # E step
        responsibilities = np.zeros(self.k)
        for i in range(self.k):
            responsibilities[i] = self.coefficients[i] * self.normal.pdf(point, mean=self.means[i], cov=self.covariances[i])
        # prevents numerical instability
        responsibilities += 1e-5
        responsibilities /= sum(responsibilities)

        # updates the weighted CS statistics with gamma
        coefficients = np.ones(self.k)
        cs_means = np.zeros((self.k, self.dim))
        cs_squares = np.zeros((self.k, self.dim, self.dim))
        for i in range(self.k):
            coefficients[i] = (1 - gamma) * self.coefficients[i] + gamma * responsibilities[i]
            cs_means[i] = (1 - gamma) * self.cs_means[i] + gamma * responsibilities[i] * point
            cs_squares[i] = (1 - gamma) * self.cs_squares[i] + gamma * responsibilities[i] * np.matmul(point[:, np.newaxis], point[:, np.newaxis].T)
        return coefficients, cs_means, cs_squares


    def online_EM(self, states: List[np.ndarray], gamma: float, permute: bool = False):
        if permute:
            states = self.rng.permutation(states)

        for j in range(len(states) - 1):
            state = states[j]

            coefficients, cs_means, cs_squares = self.update_cs_statistics(state, gamma)
            self.coefficients = copy.deepcopy(coefficients) # new mixing coefficients
            for i in range(self.k):
                self.means[i] = cs_means[i] / coefficients[i]
                sigma = (cs_squares[i] - np.matmul(self.means[i][:, np.newaxis], cs_means[i][:, np.newaxis].T)) / coefficients[i]
                # prevents numerical instability
                self.covariances[i] = modify_covariance_matrix(sigma)

            self.cs_means = copy.deepcopy(cs_means)
            self.cs_squares = copy.deepcopy(cs_squares)


    def save(self, filepath: str):
        filepath = filepath.split('.json')[0]
        # attributes
        configuration = {p: p for p in [self.k, self.random_seed]}
        # along with the initial random seed and the states of the two random generators
        configuration['random_state'] = self.rng.bit_generator.state
        configuration['normal_state'] = self.normal.random_state.bit_generator.state
        # CS statistics and GMM's parameters
        configuration['mixing_coefficients'] = self.coefficients.tolist()
        configuration['means'] = self.means.tolist()
        configuration['covariances'] = self.covariances.tolist()
        configuration['cs_means'] = self.cs_means.tolist()
        configuration['cs_squares'] = self.cs_squares.tolist()

        with open(filepath + '_config.json', 'w') as f:
            f.write(json.dumps(configuration))


    def log_likelihood(self, data: np.ndarray):
        if len(data.shape) == 1:
            return np.log(sum([self.coefficients[i] * self.normal.pdf(data, mean=self.means[i], cov=self.covariances[i]) for i in range(self.k)]))
        # data is an array of points
        else:
            log_likelihoods = []
            for x in data:
                tmp = 0.0
                for i in range(self.k):
                    tmp += self.coefficients[i] * self.normal.pdf(x, mean=self.means[i], cov=self.covariances[i])
                log_likelihoods.append(np.log(tmp))
            return sum(log_likelihoods)

    def gmm(self, state: np.ndarray):
        total = 0.0
        for i in range(self.k):
            total += self.coefficients[i] * self.normal.pdf(state, mean=self.means[i], cov=self.covariances[i])
        return total



class CoverageModel():
    def __init__(self, random_seed: int, k: int, gamma: float) -> None:
        assert k > 0
        assert gamma > 0.0 and gamma < 1.0

        self.k = k
        self.gamma = gamma

        # random generators
        self.random_seed = random_seed
        self.rng: np.random.Generator = np.random.default_rng(self.random_seed)
        self.normal = multivariate_normal
        self.normal.random_state = np.random.default_rng(self.random_seed)

        self.GMM_s = GMM(self.random_seed, k)
        self.GMM_c = GMM(self.random_seed, k)


    def initialize(self, states: List[np.ndarray]):
        '''Initializes the GMMs\' parameters and the CS statistics.'''
        assert len(states) == self.k + 1

        self.GMM_s.initialize(np.array(states))
        states_concatenated = np.array([np.hstack([states[i], states[i+1]] for i in range(self.k))])
        print(f'concatenated states of shape {states_concatenated.shape}')
        self.GMM_c.initialize(states_concatenated)



    def sequence_freshness(self, state_sequence: List[np.ndarray], tau: float):
        first_state = state_sequence[0]
        density = self.GMM_s.gmm(first_state)

        # indices are shifted compared to the definition
        for i in range(1, len(state_sequence)):
            density *= (self.GMM_s.gmm(state_sequence[i]) / self.GMM_c.gmm(np.hstack([state_sequence[i-1], state_sequence[i]])))

        if density < tau:
            print('updating parameters...')
            self.dynamic_EM(state_sequence)


    def dynamic_EM(self, state_sequence: List[np.ndarray]):
        states_concatenated = np.array([np.hstack([state_sequence[i], state_sequence[i+1]] for i in range(len(state_sequence)))])

        self.GMM_s.online_EM(state_sequence, self.gamma)
        self.GMM_c.online_EM(states_concatenated, self.gamma)


    def save(self, filepath: str, parameters: List[Dict] = []):
        filepath = filepath.split('.json')[0]
        self.GMM_s.save(filepath + 's')
        self.GMM_c.save(filepath + 'c')




if __name__ == '__main__':
    test_rng: np.random.Generator = np.random.default_rng(0)
    k = 2
    dim = 2
    cluster_means = [
        [1, 1],
        [4, 4],
    ]
    initial_data, fig, ax = generate_clustered_data(dim, k, cluster_means, num_points_per_cluster=1000, plot=True, spread_factor=0.05)
    shuffle_data = test_rng.permutation(initial_data)
    gmm_init_data = shuffle_data[:k]
    gmm = GMM(0, 2)
    gmm.initialize(gmm_init_data)
    plot_gaussians(gmm.means, gmm.covariances, ax, cmap_values=[0.42, 0.69])
    ax.set_xlim(ax.get_xlim())
    ax.set_ylim(ax.get_ylim())

    i = 0
    fig.savefig(f'imgs/iteration_{i}.png')

    num_iterations = 10
    batch_size = 100
    gamma = 0.01
    ll = [gmm.log_likelihood(shuffle_data)]
    for _ in tqdm.tqdm(range(num_iterations)):
        samples = shuffle_data[test_rng.choice(len(shuffle_data), size=batch_size)]
        gmm.online_EM(samples, gamma=gamma)
        ll.append(gmm.log_likelihood(shuffle_data))

        i += 1
        remove_gaussian(ax)
        plot_gaussians(gmm.means, gmm.covariances, ax, cmap_values=[0.42, 0.69])
        fig.savefig(f'imgs/iteration_{i}.png')

    plot_gaussians(gmm.means, gmm.covariances, ax)
    fig.savefig('test_gmm.png')
    create_gif('imgs', 'test_gmm.gif', duration=350)