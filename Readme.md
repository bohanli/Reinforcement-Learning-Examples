# Reinforcement-Learning-Examples

Toy programs for understanding Reinforcement Learning.

## Introduction

> Get familiar with the fundamental elements of RL: 
> - Policy, reward, value, and a model of the environment

Examples: 
- 1D_World

## Multi-Arm Bandit
> Multi-Arm Bandit is a simplifed RL problem, where there are no environmental states. If actions are allowed to affect the next situation as well as the reward, then we have the full reinforcement learning problem.

Examples: 
- Multi_Arm_Bandit

## Finite Markov Decision Processes
> Finite MDPs are all you need to understand 90% of modern reinforcement learning. The Bellman equation is the core.

## Dynamic Programming
> DP requires the complete probability distributions of all possible transitions.
> Policy Iteration: Policy Evaluation + Policy Improvement.

Examples: 
- Car_Rental

## Monte Carlo Methods
> Monte Carlo methods require only experience – sample sequences of states, actions, and rewards from actual or simulated interaction with an environment. Unlike the previous chapter, here we do not assume complete knowledge of the environment.

Examples: 
- Tic_Tac_Toe

## Temporal-Difference Learning
> TD learning is a combination of Monte Carlo ideas and dynamic programming (DP) ideas.
> - Like Monte Carlo methods, TD methods can learn directly from raw experience without a model of the environment’s dynamics.
> - Like DP, TD methods update estimates based in part on other learned estimates, without waiting for a final outcome (they bootstrap).
> 
> The two examples can help you get familiar with two kinds of TD control methods
> - SARSA: On-Policy TD Control
> - Q-learning: Off-Policy TD Control

Examples: 
- Windy_Grid_World
- Cliff_Walking

---
# Reference
> Sutton & Barto's book Reinforcement Learning: An Introduction (2nd Edition)
