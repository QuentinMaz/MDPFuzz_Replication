import numpy as np
import pytest

from mdpfuzz.gmm import GMM
from mdpfuzz.utils import (
    create_gif,
    generate_clustered_data,
    plot_gaussians,
    remove_gaussian,
)


@pytest.fixture(scope="module")
def input_data():
    k = dim = 2
    gamma = 0.01
    means = [[1, 1], [4, 4]]
    cmap_values = [0.25, 0.55]
    test_rng = np.random.default_rng(42)  # type: np.random.Generator
    initial_data, fig, ax = generate_clustered_data(
        dim,
        k,
        means,
        num_points_per_cluster=1000,
        plot=True,
        spread_factor=0.05,
        rng=test_rng,
    )
    gmm = GMM(0, k)
    gmm.set_gamma(gamma)
    test_data = {
        "k": k,
        "dim": dim,
        "gamma": 0.01,
        "gmm": gmm,
        "means": means,
        "rng": test_rng,
        "initial_data": initial_data,
        "fig": fig,
        "ax": ax,
        "colors": cmap_values,
    }

    return test_data


def test_initialize(input_data):
    rng = input_data["rng"]
    data_shuffled = rng.permutation(input_data["initial_data"])
    gmm = input_data["gmm"]  # type: GMM
    gmm.initialize(data_shuffled[: input_data["k"]])

    assert True

    # dirty
    fig = input_data["fig"]
    ax = fig.get_axes()[0]
    plot_gaussians(gmm.means, gmm.covariances, ax, cmap_values=input_data["colors"])
    ax.set_xlim(ax.get_xlim())
    ax.set_ylim(ax.get_ylim())
    fig.savefig("imgs/test_initialize.png")


def test_online_gmm(input_data):
    rng = input_data["rng"]
    data_shuffled = rng.permutation(input_data["initial_data"])
    gmm = input_data["gmm"]  # type: GMM
    ll = [gmm.log_likelihood(data_shuffled)]

    m = len(data_shuffled)
    batch_size = 100

    fig = input_data["fig"]
    ax = fig.get_axes()[0]

    for i in range(10):
        samples = data_shuffled[rng.choice(m, size=batch_size)]
        before = gmm.log_likelihood(samples)
        gmm.online_EM(samples)
        after = gmm.log_likelihood(samples)
        assert before < after
        ll.append(gmm.log_likelihood(data_shuffled))
        remove_gaussian(ax)
        plot_gaussians(gmm.means, gmm.covariances, ax, cmap_values=input_data["colors"])
        fig.savefig("imgs/iteration_{:02d}.png".format(i))

    # dirty
    remove_gaussian(ax)
    plot_gaussians(gmm.means, gmm.covariances, ax, cmap_values=input_data["colors"])
    fig.savefig("imgs/test_online_gmm.png")
    create_gif("imgs", "imgs/test_online_gmm.gif", duration=400, prefix="iteration")
