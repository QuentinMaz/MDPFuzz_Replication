import time
from typing import Any, List, Tuple

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from gymnasium.wrappers.time_limit import TimeLimit
from matplotlib.ticker import MaxNLocator
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback

from mdpfuzz.executor import Executor
from mdpfuzz.logger import FuzzerLogger
from mdpfuzz.mdpfuzz import Fuzzer

MODEL_PATH = "dqn_mountain_car.zip"


class StopOnFailureRateCallback(BaseCallback):
    """
    Callback for stopping training once the agent has reached a specific failure rate threshold.

    Args:
        eval_env (gym.Env): The environment used for evaluation.
        eval_frequency (int): Frequency of the callback.
        num_eval_episodes (int, optional): The number of episodes to approximate the failure rate of the agent. Default to 100.
        failure_rate_treshold (float, optional): Failure rate threshold to stop training. Default to 0.10.
        start_on_steps (int, optional): Steps after which the callback will be called. Defaults to 0.
    """

    def __init__(
        self,
        eval_env: gym.Env,
        eval_frequency: int,
        num_eval_episodes=int,
        failure_rate_treshold=0.10,
        start_on_steps=0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.eval_env = eval_env
        self.eval_frequency = eval_frequency
        self.num_eval_episodes = num_eval_episodes
        self.failure_rate_treshold = failure_rate_treshold
        self.start_on_steps = start_on_steps

    def _on_step(self):

        if self.n_calls >= self.start_on_steps and (
            self.n_calls % self.eval_frequency == 0
        ):
            failure_rate = self._evaluate_failure_rate()
            if self.verbose > 0:
                print(
                    "Failure rate at steps {}: {:.2%}.".format(
                        self.n_calls, failure_rate
                    )
                )
            if failure_rate <= self.failure_rate_treshold:
                return False
        return True

    def _evaluate_failure_rate(self) -> float:
        num_failures = 0
        for _ in range(self.num_eval_episodes):
            obs, info = self.eval_env.reset()
            terminated = truncated = False
            state = False
            while not (terminated or truncated):
                action, state = self.model.predict(obs, state=state, deterministic=True)
                obs, reward, terminated, truncated, info = self.eval_env.step(action)
            num_failures += int(terminated == False)
        failure_rate = num_failures / self.num_eval_episodes
        return failure_rate


def train_policy():
    """
    Trains a DQN policy for Mountain Car.
    Hyperparameters are borrowed from:
    https://github.com/Stable-Baselines-Team/rl-colab-notebooks/blob/sb3/dqn_sb3.ipynb
    """
    env = gym.make("MountainCar-v0", render_mode="rgb_array")
    dqn_model = DQN(
        "MlpPolicy",
        env,
        verbose=1,
        train_freq=16,
        gradient_steps=8,
        gamma=0.99,
        exploration_fraction=0.2,
        exploration_final_eps=0.07,
        target_update_interval=600,
        learning_starts=1000,
        buffer_size=10000,
        batch_size=128,
        learning_rate=4e-3,
        policy_kwargs=dict(net_arch=[256, 256]),
        seed=2,
    )

    callback = StopOnFailureRateCallback(
        eval_env=gym.make("MountainCar-v0", render_mode="rgb_array"),
        eval_frequency=500,
        num_eval_episodes=100,
        failure_rate_treshold=0.10,
        verbose=1,
        start_on_steps=82_000,
    )

    dqn_model.learn(total_timesteps=90_000, callback=callback)
    print("Training stops after {} steps.".format(callback.n_calls))
    dqn_model.save(MODEL_PATH)


def accumulate_failures(failures: np.ndarray) -> np.ndarray:
    if failures.dtype != int:
        failures = failures.astype(int)

    num_failures = 0
    acc_failures = []
    for f in failures:
        num_failures += f
        acc_failures.append(num_failures)
    return np.array(acc_failures, dtype=int)


class MountainCarExecutor(Executor):

    def __init__(self, sim_steps, env_seed) -> None:
        super().__init__(sim_steps, env_seed)
        self.env = gym.make(
            "MountainCar-v0", render_mode="rgb_array"
        )  # type: TimeLimit
        # from MountainCar-v0 specifications:
        # lower and upper bounds of the initial x position of the car
        self.low = -0.6
        self.high = -0.4
        self.mutation_intensity = 0.05

    def generate_input(self, rng: np.random.Generator) -> np.ndarray:
        # from the original implementation
        return np.array(
            [rng.uniform(low=self.low, high=self.high), 0.0], dtype=np.float32
        )

    def generate_inputs(self, rng: np.random.Generator, n: int) -> np.ndarray:
        if n == 1:
            return self.generate_input(rng)
        else:
            return np.vstack(
                [self.generate_input(rng) for _ in range(n)], dtype=np.float32
            )

    def mutate(
        self, input: np.ndarray, rng: np.random.Generator, **kwargs
    ) -> np.ndarray:
        position, velocity = input
        new_position = np.clip(
            rng.normal(position, self.mutation_intensity), self.low, self.high
        )
        return np.array([new_position, velocity], dtype=np.float32)

    def load_policy(self, **kwargs):
        model_path = kwargs["model_path"]
        return DQN.load(model_path)

    def execute_policy(
        self, input: np.ndarray, policy: DQN
    ) -> Tuple[float, bool, np.ndarray, float]:
        t0 = time.time()
        obs, _info = self.env.reset(seed=self.env_seed)

        # sets the position and velocity of the car w.r.t `input`
        position, velocity = input
        self.env.unwrapped.state = (position, velocity)
        obs = np.array(self.env.unwrapped.state, dtype=np.float32)

        acc_reward = 0.0
        state = None
        obs_seq = []
        while True:
            obs_seq.append(obs)
            action, state = policy.predict(obs, state=state, deterministic=True)
            obs, reward, terminated, truncated, _info = self.env.step(action)
            acc_reward += reward
            if terminated or truncated:
                break

        return (
            acc_reward,
            not terminated,
            np.array(obs_seq, dtype=np.float32),
            time.time() - t0,
        )


if __name__ == "__main__":
    # trains a DQN policy and saves the model
    train_policy()

    # creates an Executor object for testing the policy
    executor = MountainCarExecutor(sim_steps=200, env_seed=0)
    policy = executor.load_policy(model_path=MODEL_PATH)

    # sets a testing budget and tries MDPFuzz and Random Testing!
    testing_budget = 2500
    fuzzer = Fuzzer(random_seed=0, k=4, tau=0.1, gamma=0.01, executor=executor)
    fuzzer.random_testing(
        n=testing_budget,
        policy=policy,
        path="random_testing",
        local_sensitivity=True,
        exp_name="Mountain Car",
    )

    del fuzzer
    fuzzer = Fuzzer(random_seed=0, k=4, tau=0.1, gamma=0.01, executor=executor)
    fuzzer.fuzzing(
        n=500,
        test_budget=testing_budget,
        policy=policy,
        saving_path="mdpfuzz",
        local_sensitivity=True,
        exp_name="Mountain Car",
        save_logs_only=True,
    )

    # compares the number of faults found by the two methods
    rt_failures = accumulate_failures(
        FuzzerLogger("random_testing_logs.txt")
        .load_logs()["oracle"]
        .astype(int)
        .to_numpy()
    )
    mdpfuzz_failures = accumulate_failures(
        FuzzerLogger("mdpfuzz_logs.txt").load_logs()["oracle"].astype(int).to_numpy()
    )

    assert len(rt_failures) == len(mdpfuzz_failures)

    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(rt_failures))
    ax.plot(x, rt_failures, label="Random Testing")
    ax.plot(x, mdpfuzz_failures, label="MDPFuzz")
    ax.legend()
    ax.set_xlabel("# Iterations")
    ax.set_ylabel("# Failures")
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_title("Number of failures over test iterations")
    fig.tight_layout()
    fig.savefig("failure_results_comparison_demo.png")
