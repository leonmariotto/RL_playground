#!/usr/bin/env python3
import click
import numpy as np
import torch
import gymnasium as gym
from matplotlib import pyplot as plt
from utils.progress_bar import print_progress_bar

L1 = 4  # A
L2 = 150
L3 = 2  # B

GAMMA = 0.99
LEARNING_RATE = 0.009


class BasicPGM:
    """
    A basic policy gradient method.
    The model produce a ditributed probability over actions.
    The approach here is to run an entire episode and update the model at the end of the episode.
    This is the REINFORCE algorithm.
    """

    def __init__(self, render: bool = False):
        self.model = torch.nn.Sequential(
            torch.nn.Linear(L1, L2),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(L2, L3),
            torch.nn.Softmax(dim=0),  # C
        )
        self.optimizer = torch.optim.Adam(self.model.parameters(), LEARNING_RATE)
        if render:
            self.env = gym.make("CartPole-v1", render_mode="human")
        else:
            self.env = gym.make("CartPole-v1")
        self.score = []

    @staticmethod
    def running_mean(x, N=50):
        kernel = np.ones(N)
        conv_len = x.shape[0] - N
        y = np.zeros(conv_len)
        for i in range(conv_len):
            y[i] = kernel @ x[i : i + N]
            y[i] /= N
        return y

    @staticmethod
    def discount_rewards(rewards):
        """
        Take a vector of rewards in parameters and return vector of discounted rewards.
        Use GAMMA meta parameter.
        Normalize value.
        """
        lenr = len(rewards)
        disc_return = torch.pow(GAMMA, torch.arange(lenr).float()) * rewards  # A
        disc_return /= disc_return.max()  # B
        return disc_return

    @staticmethod
    def loss_fn(preds, r):  # A
        """
        parameter r is a vector of discount factor.
        preds is all the prediction done by the model for this episode.
        """
        return -1 * torch.sum(r * torch.log(preds))  # B

    def run_episode(self, episode_duration: int = 200, render=False):
        """
        Run an episode.
        """
        curr_state = self.env.reset()[0]
        done = False
        transitions = []  # B
        for t in range(episode_duration):  # C
            act_prob = self.model(torch.from_numpy(curr_state).float())  # D
            action = np.random.choice(np.array([0, 1]), p=act_prob.data.numpy())  # E
            prev_state = curr_state
            curr_state, _, done, _, info = self.env.step(action)  # F
            transitions.append((prev_state, action, t + 1))  # G
            if render:
                self.env.render()
            if done:  # H
                break
        ep_len = len(transitions)  # I
        self.score.append(ep_len)
        reward_batch = torch.Tensor([r for (s, a, r) in transitions]).flip(
            dims=(0,)
        )  # J
        disc_returns = BasicPGM.discount_rewards(reward_batch)  # K
        state_batch = torch.Tensor(np.array([s for (s, a, r) in transitions]))  # L
        action_batch = torch.Tensor([a for (s, a, r) in transitions])  # M
        pred_batch = self.model(state_batch)  # N
        prob_batch = pred_batch.gather(
            dim=1, index=action_batch.long().view(-1, 1)
        ).squeeze()  # O
        loss = BasicPGM.loss_fn(prob_batch, disc_returns)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def show_metrics(self):
        score = np.array(self.score)
        avg_score = BasicPGM.running_mean(score, 50)
        plt.figure(figsize=(10, 7))
        plt.ylabel("Episode Duration", fontsize=22)
        plt.xlabel("Training Epochs", fontsize=22)
        plt.plot(avg_score, color="green")
        plt.show()

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


@click.command()
@click.option(
    "--render",
    is_flag=True,
    default=False,
    show_default=True,
    help="Render game, to use on a trained agent with -i.",
)
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
    "--episode_nb",
    "-n",
    type=int,
    default=500,
    help="Number of episodes.",
)
def run_agent(
    render: bool,
    out_weight: str,
    in_weight: str,
    episode_nb: int,
):
    basic_pgm = BasicPGM(render)
    if in_weight != "":
        basic_pgm.import_weight(in_weight)
    for i in range(episode_nb):  # C
        basic_pgm.run_episode()
        print_progress_bar(0, i, episode_nb - 1)
    if out_weight != "":
        basic_pgm.export_weight(out_weight)
    if render is False:
        basic_pgm.show_metrics()


if __name__ == "__main__":
    run_agent()
