#!/usr/bin/env python3

"""
Distributed advantage actor-critic (DA2C)
"""

import click

import numpy as np
import torch
import gymnasium as gym
from matplotlib import pyplot as plt
import torch.multiprocessing as mp

L1 = 4  # A
L2 = 150
L3 = 2  # B
GAMMA = 0.99
LEARNING_RATE = 0.009
EPISODE_TIMEOUT = 500  # 500 step max per episode


class ActorCritic(torch.nn.Module):
    """
    A custom module.
    """

    def __init__(self):
        """
        Init layers and activations functions.
        """
        super(ActorCritic, self).__init__()
        self.l1 = torch.nn.Linear(4, 25)
        self.l2 = torch.nn.Linear(25, 50)
        self.actor_lin1 = torch.nn.Linear(50, 2)
        self.l3 = torch.nn.Linear(50, 25)
        self.critic_lin1 = torch.nn.Linear(25, 1)

    def forward(self, x):
        """
        Apply neural network operations.
        In -> Linear(4, 25) -> ReLU -> Linear(25, 50) -> ReLU -> Linear(50, 2) -> LogSoftMax -> Actor
        In -> Linear(4, 25) -> ReLU -> Linear(25, 50) -> ReLU -> Linear(50, 2) -> ReLU -> Tanh -> Critic
        """
        x = torch.nn.functional.normalize(x, dim=0)
        y = torch.nn.functional.relu(self.l1(x))
        y = torch.nn.functional.relu(self.l2(y))
        actor = torch.nn.functional.log_softmax(self.actor_lin1(y), dim=0)
        c = torch.nn.functional.relu(self.l3(y.detach()))
        critic = torch.tanh(self.critic_lin1(c))
        return actor, critic


class DA2C:
    """
    Distributed advantage actor-critic (DA2C)
    Actor critic model combine a Q-learner and a policy learner.
    """

    def __init__(self):
        self.model = ActorCritic()
        self.optimizer = torch.optim.Adam(self.model.parameters(), LEARNING_RATE)
        self.env = gym.make("CartPole-v1")
        self.score = []

    def run_episode(self):
        """
        Run an episode.
        For each step compute policy and value from state.
        Compute the action from the policy:
            - logits are unnormalized action scores.
        """
        self.optimizer.zero_grad()
        state, info = self.env.reset()
        state = torch.from_numpy(state).float()
        values, logprobs, rewards = [], [], []
        done = False
        j = 0
        while not done:  # C
            j += 1
            policy, value = self.model(state)
            values.append(value)  # log the critic network value
            logits = policy.view(-1)  # reshapes the tensor policy into a flat 1D vector
            action_dist = torch.distributions.Categorical(
                logits=logits
            )  # create a probability distribution
            action = (
                action_dist.sample()
            )  # sample an action from the probability distribution
            logprob_ = policy.view(-1)[action]
            logprobs.append(logprob_)  # log the probability of the chosen action
            state_, _, done, _, info = self.env.step(action.detach().numpy())
            state = torch.from_numpy(state_).float()
            if done:
                reward = -10
                self.env.reset()
            else:
                reward = 1.0
            rewards.append(reward)  # log rewards
            if j > EPISODE_TIMEOUT:
                done = True
                self.env.reset()
        # Output critic model values, probability of taken action, and rewards.
        return values, logprobs, rewards

    def update_params(self, values, logprobs, rewards, clc=0.1, gamma=0.95):
        """
        update parameters with episodes values.
        compute an overall loss with actor_loss and critic_loss.
        """
        rewards = torch.Tensor(rewards).flip(dims=(0,)).view(-1)
        logprobs = torch.stack(logprobs).flip(dims=(0,)).view(-1)
        values = torch.stack(values).flip(dims=(0,)).view(-1)
        returns = []
        ret_ = torch.Tensor([0])
        for r in range(rewards.shape[0]):
            ret_ = rewards[r] + gamma * ret_
            returns.append(ret_)
        returns = torch.stack(returns).view(-1)
        returns = torch.nn.functional.normalize(returns, dim=0)
        actor_loss = -1 * logprobs * (returns - values.detach())
        critic_loss = torch.pow(values - returns, 2)
        loss = actor_loss.sum() + clc * critic_loss.sum()
        loss.backward()
        self.optimizer.step()
        # Return list of actor loss, list of critic loss and length of episode.
        return actor_loss, critic_loss, len(rewards)

    def export_weight(self, filepath: str):
        """
        Export model weight in filepath.
        """
        torch.save(self.model.state_dict(), filepath)

    def import_weight(self, filepath: str):
        """
        Import weigth from a file.
        Model must match the nn architecture.
        """
        self.model.load_state_dict(torch.load(filepath))


def worker(i: int, da2c: DA2C, counter, epoch_nb: int, queue):
    """ """
    metrics = {
        "actor_loss": [],
        "critic_loss": [],
        "eplen": [],
    }
    for _ in range(epoch_nb):
        values, logprobs, rewards = da2c.run_episode()  # B
        actor_loss, critic_loss, eplen = da2c.update_params(
            values, logprobs, rewards
        )  # C
        metrics["actor_loss"] += [actor_loss.detach().cpu().numpy()]
        metrics["critic_loss"] += [critic_loss.detach().cpu().numpy()]
        metrics["eplen"] += [eplen]
        counter.value = counter.value + 1  # D
    queue.put(metrics)


def display_metrics(metrics):
    fig, axs = plt.subplots(3, 1, figsize=(10, 18))
    for metric in metrics:
        actor_loss = np.concatenate(metric["actor_loss"])
        critic_loss = np.concatenate(metric["critic_loss"])
        axs[0].set_title("actor_loss [0]", fontsize=22)
        axs[0].set_xlabel("Steps", fontsize=16)
        axs[0].set_ylabel("Loss", fontsize=16)
        axs[0].plot(actor_loss)
        axs[1].set_title("critic_loss [0]", fontsize=22)
        axs[1].set_xlabel("Steps", fontsize=16)
        axs[1].set_ylabel("Loss", fontsize=16)
        axs[1].plot(critic_loss)
        axs[2].set_title("eplen [0]", fontsize=22)
        axs[2].set_xlabel("Steps", fontsize=16)
        axs[2].set_ylabel("Loss", fontsize=16)
        axs[2].plot(metric["eplen"])
    plt.tight_layout(pad=5.0)
    plt.show()


def run_training(da2c: DA2C, worker_nb: int = 7, epoch_nb: int = 200):
    queue = mp.Queue()
    processes = []
    metrics = []
    da2c.model.share_memory()
    counter = mp.Value("i", 0)
    for i in range(worker_nb):
        p = mp.Process(target=worker, args=(i, da2c, counter, epoch_nb, queue))
        p.start()
        processes.append(p)

    for _ in range(worker_nb):
        metrics += [queue.get()]

    for p in processes:
        p.join()
    for p in processes:
        p.terminate()
    display_metrics(metrics)


@click.command()
@click.option(
    "--out_weight",
    "-o",
    type=str,
    default="",
    help="Filepath to output weigth file.",
)
@click.option(
    "--in_weight",
    "-i",
    type=str,
    default="",
    help="Filepath to input weigth file.",
)
@click.option(
    "--train",
    is_flag=True,
    default=False,
    show_default=True,
    help="Train model.",
)
@click.option(
    "--proc",
    "-n",
    type=int,
    default=8,
    show_default=True,
    help="Number of thread used to train.",
)
def run_agent(
    out_weight: str,
    in_weight: str,
    train: bool,
    proc: int,
):
    da2c = DA2C()
    if in_weight != "":
        da2c.import_weight(in_weight)
    if train is True:
        run_training(da2c, proc)
    if out_weight != "":
        da2c.export_weight(out_weight)


if __name__ == "__main__":
    run_agent()
