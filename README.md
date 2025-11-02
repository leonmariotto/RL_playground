# RL playground

Multi-project repository to play around with reinforcement learning.
Beat some OpenAI Gym games.

## Algorithms

Reinforcement Learning algorithm.

### REINFORCE

The `basic_pgm.py` implement a basic policy gradient method called REINFORCE.

### DA2C

Distributed advantage actor-critic (DA2C) algorithm.
This is a more advanced model. Actor critic model combine a Q-learner and a policy learner.
Use multi-threading for training.

## TODO

- beat Pong with da2c.
- make agent more independant of env, and put them in class, to avoid code repetition.
