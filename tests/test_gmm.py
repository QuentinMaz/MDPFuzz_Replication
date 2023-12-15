import pytest
import copy
import numpy as np

import src.gmm as gmm
from src.utils import plot_gaussians, generate_clustered_data


@pytest.fixture(scope='module')
def input_data():
    k = dim = 2
    means = [[1, 1], [4, 4]]
    test_rng: np.random.Generator = np.random.default_rng(42)
    initial_data, fig, ax = generate_clustered_data(
        dim,
        k,
        means,
        num_points_per_cluster=1000,
        plot=True,
        spread_factor=0.05
    )
    gmm = gmm.GMM(0, k)
    test_data = {
        'k': k,
        'dim': dim,
        'gamma': 0.01,
        'gmm': gmm,
        'means': means,
        'rng': test_rng,
        'initial_data': initial_data,
        'fig': fig,
        'ax': ax
    }

    return test_data



def test_initialize(input_data):
    rng = input_data['rng']
    data_shuffled = rng.permutation(input_data['initial_data'])
    gmm: gmm.GMM = input_data['gmm']
    gmm.initialize(data_shuffled[:input_data['k']])

    assert True

    # dirty
    fig = input_data['fig']
    ax = fig.get_axes()[0]
    plot_gaussians(gmm.means, gmm.covariances, ax)
    fig.savefig('imgs/test_initialize.png')



def test_online_gmm(input_data):
    rng = input_data['rng']
    data_shuffled = rng.permutation(input_data['initial_data'])
    gmm: gmm.GMM = input_data['gmm']
    ll = [gmm.log_likelihood(data_shuffled)]

    m = len(data_shuffled)
    gamma = input_data['gamma']
    batch_size = 100

    for _ in range(10):
        samples = data_shuffled[rng.choice(m, size=batch_size)]
        before = gmm.log_likelihood(samples)
        gmm.online_gmm(samples, gamma=gamma)
        after = gmm.log_likelihood(samples)
        assert before < after
        ll.append(gmm.log_likelihood(data_shuffled))

    # dirty
    fig = input_data['fig']
    ax = fig.get_axes()[0]
    plot_gaussians(gmm.means, gmm.covariances, ax)
    fig.savefig('imgs/test_online_gmm.png')




# @pytest.fixture()
# def plot_after_test(request):
#     yield
#     test_name = request.node.name.replace('test_', '')


# def plot_test_gaussians(fig, means, covariances, filename):
#     ax = fig.get_axes()[0]
#     plot_gaussians(means, covariances, ax)
#     fig.savefig(f'imgs/{filename}.png')


# def test_and_plot():
#     means, covariances = test_initialize()
#     plot_test_gaussians(fig, means, covariances, 'test_initialize')


'''
import pytest
import matplotlib.pyplot as plt

# Fixture to create objects/data for tests
@pytest.fixture
def test_data():
    # Generate or create necessary test data
    return [1, 2, 3], [4, 5, 6]  # Example data

# Test function using the fixture
def test_example(test_data):
    # Perform assertions or verifications
    assert len(test_data[0]) == len(test_data[1])

    # Return data for potential plotting
    return test_data

# Function to plot based on returned test data
def plot_test_data(test_data):
    plt.plot(test_data[0], test_data[1])
    plt.title("Plot using test data")
    plt.savefig(f'path_to_save/test_plot.png')
    plt.close()

# Example usage of the test and plotting functions
def test_and_plot():
    # Run the test function and obtain data
    data = test_example()

    # Plot based on the test data
    plot_test_data(data)
'''