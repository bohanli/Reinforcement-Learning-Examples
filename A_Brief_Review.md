# A Brief Review on the RLAI Book

Oct  10, 2017

---

Reinforcement Learning is about learning from interaction. **A learning agent interacts with an uncertain environment to achieve a goal (long term return)**.

Chapter 1 gives a introduction to the concept of RL.
- The **characteristics** of RL,
- The **difference** between RL and traditional machine learning problems (supervised/unsupervised learning)
- The basic **elements** of RL, including (1) policy, (2) reward, (3) value, (4) a model of the environment
- Relationship with **evolutionary methods**

---

These books contains two parts.
- Part 1 (Chapter 2 - Chapter 8) focuses on tabular methods without function approximation.
- Part 2 (Chapter 9 - Chapter 13), focuses on function approximation.

---
# Part I

Chapter 2 -- **Multi-Arm Bandit**.
- Multi-Arm Bandit is a simplifed RL problem, where there are no environmental states. If actions are allowed to affect the next situation as well as the reward, then we have the full reinforcement learning problem.

Chapter 3 -- **Markov Decision Process**

Finite MDPs are all you need to understand 90% of modern RL. The Bellman equation is the core.

- This chapter first summarize the elements of the RL problem in a more systematic way.
    - The RL **agent** and its **environment** interact over a sequence of discrete time steps.
        - The agent–environment boundary <-- determined by agent’s absolute control, not of its knowledge
    - The agent-environment interface
        - **actions**: the choices made by the agent <--- given by a policy (a function of state)
        - **states**: the basis for **making the choices**
        - **rewards**: the basis for **evaluating the choices**
    - Goals: **return** (function of future **rewards**, formulation is different for **episodic** and **continuing** tasks)
- Then comes the most important concept -- **Markov Decision Process**. Most of the current theory of RL is restricted to finite MDPs.
    - **value functions**
        - the **expected return** from that state, or state–action pair given a **policy**
        - **Bellman equation**:
            $$ v_\pi (s) = E_\pi(G_t|s_t=s) = \sum_a \pi(a|s) \sum_{s', r} p(s', r|s, a)[r+\gamma v_\pi(s')]  $$
        - optimal policy: A policy whose value functions are optimal

---

In general, there are two types of methods in RL
1. planning methods: methods that **require a model** of the environment, such as
    - Dynamic programming (Chapter 4)
    - Heuristic search (Chapter 8)
2. learning methods: methods that can be used **without a model**, such as
    - Monte Carlo methods (Chapter 6)
    - Temporal-difference methods (Chapter 7)

For each type of methods, we have two type of tasks
1. Prediction: estimating the state-value function $v_\pi$ for a given policy
2. Control: approximate optimal policies

---
Chapter 4 -- **Dynamic Programming**
The core concept of this chapter is **generalized policy iteration (GPI)**

GPI refer to the general idea of letting policy evaluation and policy improvement processes interact.
- policy evaluation: making the value function consistent with the current policy
- policy improvement: making the policy greedy with respect to the current value function

**policy iteration**: these two processes alternate, each completing before the other begins --> **value iteration**: only a single iteration of policy evaluation is performed in between each policy improvement --> **asynchronous DP methods**: the evaluation and improvement processes are interleaved at an even finer grain.

As long as both processes continue to update all states, convergence can be guaranteed



Chapter 5 -- **Monte Carlo Methods**

Monte Carlo methods do not required a model (learning methods)
1. Prediction: simply to average the returns observed after visits to that state. As more returns are observed, the average should converge to the expected value.
2. Control: Trade-off between exploration and exploiting
    - In **on-policy** methods, the agent commits to always exploring and tries to find the best policy that still explores.
        - With exploring start
        - Without exploring start: $\epsilon$-greedy
    - In **off-policy** methods, the agent also explores, but learns a deterministic optimal policy that may be unrelated to the policy followed. -->  via Importance Sampling


---

Chapter 6 -- **Temporal Difference Learning**

Temporal-difference means that TD methods **change an earlier estimate based on how it differs from a later estimate**

TD learning is a combination of Monte Carlo ideas and dynamic programming (DP) ideas.
- Like Monte Carlo methods: **without a model**
- Like DP: update estimates based in part on other learned estimates, without waiting for a final outcome (they **bootstrap**).

prediction and control of TD
1. prediction: TD methods combine the sampling of Monte Carlo with the bootstrapping of DP.
    1. it samples the expected values in $E_{\pi}[r+\gamma v_\pi(s')]$ -- Monte Carlo methods also use an estimate of $v_\pi (s) = E_\pi(G_t|s_t=s)$ as a target
    2. and it uses the current estimate $V$ instead of the true $v_{\pi}$ -- like DP

    - TD error: $\delta_t = R_{t+1} + \gamma V(S_{t+1}) - V(S_t)$
TD error is not actually available until one time step later

2. control: all use some variation of generalized policy iteration (GPI).
    - Sarsa: On-Policy TD Control
    - Q-learning: Off-Policy TD Control
    - Expected Sarsa

---

Chapter 7 -- **Multi-step Bootstrapping**

Monte Carlo methods <--- Multi-step TD ---> one-step TD methods
1. Prediction: n-step TD Prediction
2. Control:
    - n-step Sarsa
        - has all sampled transitions
    - n-step Off-policy Learning by Importance Sampling
    - n-step Off-policy Learning without Importance Sampling -- the tree-backup algorithm
        - has all state-to-action transitions fully branched without sampling
    - n-step Expected Sarsa backup
        - has all sample transitions except for the last state-to-action ones, which are fully branched with an expected value.
    - A Unifying Algorithm: n-step Q($\sigma$)
        - one might decide on a step-by-step basis whether one wanted to take the action as a sample, as in Sarsa,
        - or consider the expectation over all actions instead, as in the tree backup.
---

Chapter 8 -- **Planning and Learning with Tabular Methods**

More about planning
- The environment may change. Thus the agent should be able to update itself accordingly.
- planning can be much more effcient if simulated transitions and backups are focused on particular state–action pairs.
    - Two ways of thinking about planning.
        1. planning as the gradual improvement of a policy or value function that is **good in all states** generally rather than focused on any particular state (e.g., dynamic programming and Dyna)
        2. Planning as Part of Action Selection -- discarded after being used to select the current action
        - Heuristic Search --> Monte Carlo Tree Search

Dyna: unify planning and learning methods
---

# Part II

tabular methods --> approximation
- motivation
    1. ininite state space
    2. generalization

Chapter 9 -- **On-policy Prediction with Approximation**
- Prediction objective: MSVE
- Optimization method:
    - Stochastic-gradient (Monte Carlo)
    - Semi-gradient Methods (bootstrapping, like TD)
- Approximation function
    - Linear Methods
    - Nonlinear Function Approximation: Artificial Neural Networks
    - Least Square TD: data-efficient but computation-costing

Chapter 10 -- **On-policy Control with Approximation**
1. For Episodic tasks:
    - Semi-gradient Control --> n-step Semi-gradient Sarsa
2. For Continuing Tasks: replce discounted return with Average Reward
    - n-step Differential Semi-gradient Sarsa

Chapter 11 -- **Off-policy Methods with Approximation**
- Semi-gradient Methods with importance sampling <-- Baird’s Counterexample shows that this method cannot guarantee convergence
- The Deadly Triad: the danger of instability and divergence arises whenever we combine three things

---

Chapter 12 -- **Eligibility Traces**

The TD($\lambda$) algorithm can be understood as one particular way of averaging n-step backups. This average contains all the n-step backups, each weighted proportional to $\lambda^{n-1}$
1. $\lambda=1$ -- --> Monte Carlo algorithm
2. $\lambda=0$ -- --> $G_t^{(1)}$, a one-step TD method

TD($\lambda$) improves over the off-line $\lambda$-return algorithm in three ways
1. it updates the weight vector on every step of an episode rather than only at the end, and thus its estimates may be better sooner.
2. its computations are equally distributed in time rather that all at the end of the episode.
3. it can be applied to continuing problems rather than just episodic problems.

Disadvantage of TD($\lambda$): \
- TD($\lambda$) is more sentitive to $\alpha$

The online $\lambda$-return algorithm (-- with h-truncated $\lambda$-return ) --> True Online TD($\lambda$)

---

Chapter 13 -- **Policy Gradient Methods**

learn a parameterized policy that can select actions without consulting a value function.

The Policy Gradient Theorem --> policy gradient methods
- REINFORCE: Monte Carlo Policy Gradient --> REINFORCE with Baseline (to decrease variance)
- Actor-Critic Methods -- bootstrapping


---
