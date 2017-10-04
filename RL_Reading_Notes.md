# Reinforcement Learning

## Chapter 1 | Introduction

RL is about learning from interaction. A learning agent interacts with its environment (with uncertainty) to achieve a goal.

### Reinforcement Learning
**Three characteristics:**
1. Closed-loop
    - The learning system’s actions influence its later inputs
2. Not having direct instructions as to what actions to take
    - The learner is not told which actions to take, but instead must discover which actions yield the most reward by trying them out.
3. Actions may affect not only the immediate reward but also the next situation and, through that, all subsequent rewards

**Differences from supervised/unsupervised learning**
- RL is difference from *unsupervised learning*. reinforcement learning is trying to maximize a reward signal instead of trying to find hidden structure

**The trade-off between exploration and exploitation**
- The agent has to **exploit** what it already knows in order to obtain reward, but it also has to **explore** in order to make better action selections in the future.
- The dilemma is that neither exploration nor exploitation can be pursued exclusively without failing at the task.

RL explicitly considers the **whole problem** of a goal-directed agent interacting with an uncertain environment.

### Elements of Reinforcement Learning

1. Policy
    - A policy defines the learning agent’s way of behaving at a given time.
    - states **of the environment** -> actions.
    - policies may be stochastic.
2. Reward
    - A reward signal defines the goal in a RL problem.
    - (states of the environment, actions) -> rewards
    - The reward signal thus defines what are the good and bad events for the agent. The agent’s sole objective is to maximize the total reward it receives over the long run.
3. Value
    - The value **of a state** is the total amount of reward an agent can expect to accumulate over the future, starting from that state.
    - Relationship between **reward** and **value**
        - **Reward**: what is good in an immediate sense
        - **Value**: what is good in the long run. **Predictions of rewards**.
        - Without rewards there could be no values. Nevertheless, we seek actions that bring about states of **highest value**, not highest reward.

4. A model of the environment
    - (state, action) -> (next state, next reward)
    - **Model-based** methods and **model-free** methods
        -  model-based methods: methods for solving RL problems that use models and planning
            - Planning: deciding on a course of action by considering possible future situations before they are actually experienced.
        - model-free methods: explicitly trial-and-error learners

### Limitations and Scope

**Evolutionary methods**
- Rather than estimating value functions, evolutionary methods evaluate the “lifetime” behavior of many **non-learning** agents, each using a different policy for interacting with its environment, and select those that are able to obtain the most reward.
- When can evolutionary methods be effective?
    - If the space of policies is sufficiently small,
    - or can be structured so that good policies are common or easy to find
    - or if a lot of time is available for the search
    - or learning agent cannot accurately sense the state of its environment
- Evolutionary methods ignore much of the useful structure of the RL problem:
    - they do not use the fact that the policy they are searching for is a function from states to actions
    - they do not notice which states an individual passes through during its lifetime, or which actions it selects
- Relationship with **policy gradient methods**.
    - Policy gradient methods also does not appeal to value functions. These methods search in spaces of policies defined by a collection of numerical parameters. They estimate the directions the parameters should be adjusted in order to most rapidly improve a policy’s performance. Unlike evolutionary methods, however, they produce these estimates while the agent is interacting with its environment and so can take advantage of the details of individual behavioral interactions.

# Part I -- Tabular Solution Methods
## Chapter 2 | Multi-arm Bandits
Compared with other types of learning, RL uses training information that evaluates the actions taken rather than instructs by giving correct actions.
- evaluative feedback depends entirely on the action taken, whereas instructive feedback is independent of the action taken.

### A k-Armed Bandit Problem
$q_* = E(R_t|A_t=a)$ and its estimated value $Q_t(a)$
- Exploit: In greedy methods, you would always select the action with highest value to solve the k-armed bandit problem.
- Explore.

### Action-Value Methods

**Sample-average method**
$$Q_t(a)=\frac{\text{sum of rewards when a taken prior to t}}{\text{number of times a taken prior to t}}$$
As the denominator goes to infinity, by the law of large numbers, $Q_t(a)$ converges to $q_*(a)$.

**Action selection**
- **greedy action selection** <br/>
$A_t = argmax_a Q_t(a)$
- **$\epsilon-greedy$ method** <br/>
with small probability $\epsilon$, select randomly from amongst all the actions with equal probability independently of the action-value estimates.

### Incremental Implementation

$$NewEstimate \leftarrow OldEstimate + StepSize * [Target - OldEstimate]$$

For the simplest multi-arm bandit problem, $Q(A) \leftarrow Q(A) + \frac{1}{N(A)}[R-Q(A)]$ is equivalent to averaging all the rewards of the action in the history. (So it is still a sample-average method)

### Tracking a Nonstationary Problem
For nonstationary cases, it makes sense to weight recent rewards more heavily than long-past ones.

**Exponential, recency-weighted average**
$$Q_{n+1}(A) = Q_n(A) + \alpha[R-Q_n(A)] = (1-\alpha)^nQ_1 + \sum_i^n{\alpha(1-\alpha)^{n-1}R_i}$$

### Optimistic Initial Values

Initial action values
-  a way of supplying some prior knowledge
-  a way of encouraging exploration

In the previous bandit problem, we can set all the values very large. Thus Whichever actions are initially selected, the reward is less than the starting estimates; the learner switches to other actions. The result is that all actions are tried several times before the value estimates converge. The system does a fair amount of exploration even if greedy actions are selected all the time. (only effective on stationary problems)

### Upper-Confidence-Bound (UCB) Action Selection

$A_t = argmax_a[Q_t(a)+c\sqrt{\frac{log(t)}{N_t(a)}}]$

$N_t(a)$ denotes the number of times that action a has been selected prior to time $t$, and the number $c$ > 0 controls the degree of exploration.

Hard to be generalized
1. One diffculty is in dealing with nonstationary problems
2. Another is dealing with large state spaces, particularly function approximation as developed in Part II of this book.

### Gradient Bandit Algorithms

$$H_{t+1}(a)=H_t(a)+\alpha\frac{\partial E(R_t)}{\partial H_t(a)}$$

Details are available in the book.

### Associative Search

**Generalize the problem to the full RL**
1. **Associative Search**. More than one situation, and the goal is to learn a policy: a mapping from situations to the actions that are best in those situations.
2. If actions are allowed to affect the next situation as well as the reward, then we have the full reinforcement learning problem.

### Summary

- The $\epsilon$-greedy methods: choose randomly a small fraction of the time
- UCB methods choose deterministically but achieve exploration by subtly favoring at each step the actions that have so far received fewer samples
- Gradient bandit algorithms estimate not action values, but action preferences, and favor the more preferred actions in a graded, probabilistic manner using a soft-max distribution.
- The simple expedient of initializing estimates optimistically causes even greedy methods to explore significantly.

All of theme are far from a fully satisfactory solution to the problem of balancing exploration and exploitation.

Bayesian methods is promising to effectively turn the bandit problem into an instance of the full reinforcement learning problem.

## Chapter 3 | Finite Markov Decision Processes

### The Agent–Environment Interface

A straightforward framing of the RL problem: learning from interaction to achieve a goal.

- Agent: the learner and decision-maker.
- Environment: everything outside the agent.

At each step:
1. the agent first receive a state $S_t$ from the environment.
2. Then the agent implements a mapping (**policy**, $\pi_t(a|s)$) from states to probabilities of selecting each possible action.
3. One time step later, as a consequence of its action, the agent receives a numerical reward $R_{t+1}$, and finds itself in a new state $S_{t+1}$.
    - one signal to represent the basis on which the choices are made (the states),
    - and one signal to define the agent’s goal (the rewards)

The agent’s goal is to maximize the total amount of reward it receives over the long run.

The **boundary** between agent and environment is not often the same as the physical boundary of a robot’s or animal’s body.
- Anything that cannot be changed arbitrarily by the agent is considered to be outside of it and thus part of its environment.
- We do not assume that everything in the environment is unknown to the agent. The agent–environment boundary represents the limit of the agent’s absolute **control**, not of its **knowledge**.

### Goals and Rewards

Note:
1. The reward signal is your way of communicating to the robot **what** you want it to achieve, not **how** you want it achieved.
2. rewards are computed in the environment rather than in the agent.


### Returns

Formally, we seek to maximize the expected return, whose simplest form is the sum of the rewards.

- Episodic tasks
    - Each episode ends in a special state called the terminal state, followed by a reset to a standard starting state.
    - Even if you think of episodes as ending in different ways, such as winning and losing a game, the next episode begins independently of how the previous one ended.
    - Return: $G_t = \sum_{i=t+1}^{T}R_i$
- Continuing tasks
    - the agent–environment interaction does not break naturally into identifiable episodes, but goes on continually without limit.
    - For example, this would be the natural way to formulate a continual process-control task, or an application to a robot with a long life span.
    - Return: $G_t = \sum_{k=0}^{\infty}\gamma^k R_{t+k+1}$

### Unified Notation for Episodic and Continuing Tasks

Episodic task can be viewed as a special case of continue task.
- episode termination can be regarded as a *special absorbing* state that transitions only to itself and that generates only rewards of zero.
- Value function of the terminal state must be set as 0.

### The Markov Property

The Markov property is important in reinforcement learning because decisions and values are assumed to be a function only of the current state. In order for these to be effective and informative, the state representation must be informative.

### Markov Decision Processes

Finite MDPs are all you need to understand 90% of modern reinforcement learning.

$$p(s', r|s, a) = Pr\{ S_{t+1}=s', R_{t+1}=r|S_t=s, A_t=a \}$$

Given the dynamics as specified by the above equatioon, one can compute anything else one might want to know about the environment. For example,
- expected rewards for state–action pairs
$$ r(s, a) = E(R_{t+1}|S_t=s, A_t=a) = \sum_r r\sum_{s'} p(s', r|s, a)  $$
- the state-transition probabilities
$$ p(s'|s, a) = Pr\{ S_{t+1}=s'|S_t=s, A_t=a \} = \sum_r p(s', r|s, a)  $$
- the expected rewards for state–action–next-state triples
$$ r(s, a, s') = E(R_{t+1}|S_t=s, A_t=a, S_{t+1}=s') = \frac{\sum_r r p(s', r|s, a)}{p(s'|s, a)} $$

### Value Functions

The value of a state under a policy $\pi$
$$ v_\pi (s) = E_\pi(G_t|s_t=s) = \sum_a \pi(a|s) \sum_{s', r} p(s', r|s, a)[r+\gamma v_\pi(s')]  $$

The action-value fuction for policy $\pi$
$$ q_\pi (s, a) = E_\pi(G_t|S_t=s, A_t=a)$$

Figure 3.4: Backup diagrams for $v_\pi$ and $q_\pi$.


A fundamental property of value functions used throughout reinforcement learning and dynamic programming is that they satisfy particular recursive relationships.

The Bellman equation averages over all the possibilities, weighting each by its probability of occurring. It states that the value of the start state must equal the (discounted) value of the expected next state, plus the reward expected along the way.

### Optimal Value Functions

$$ v_*(s) = max_\pi v_\pi (s) $$
for all $s\in S$

Whereas the optimal value functions for states and state–action pairs are unique for a given MDP, there can be many optimal policies. Any policy that is greedy with respect to the optimal value functions must be an optimal policy.

###### Bellman optimality equation

According to the Bellman equation, we have
$$ v_*(s) = max_{a\in A(s)} q_{\pi^*(s, a)} = max_{a\in A(s)} \sum_{s', r} p(s', r|s, a)[r+\gamma v_*(s')] $$

then we have

$$q_*(s, a) = E(R_{t+1}+\gamma v_*(S_{t+1})|S_t=s, A_t=a) = E(R_{t+1}+\gamma max_{a'} q_{\pi^*(S_{t+1}, a')})|S_t=s, A_t=a) = \sum_{s', r} p(s', r|s, a)[r+\gamma max_{a'} q_{\pi^*(S_{t+1}, a')}]
$$

Figure 3.7: Backup diagrams for $v_*$ and $q_*$

### Optimality and Approximation

### Summary

The entire subsection is worth reading.

## Chapter 4 | Dynamic Programming

### Policy Evaluation

**Iterative policy evaluation**
$$ v_{k+1} (s) = \sum_a \pi(a|s) \sum_{s', r} p(s', r|s, a)[r+\gamma v_{k}(s')]  $$

Indeed, the sequence $\{v_k\}$ can be shown in general to converge to $v_{\pi^*}$ as $k -> \infty$ under the same conditions that guarantee the existence of $v_{\pi^*}$.

### Policy Improvement

**policy improvement theorem**
- for all $s \in S$, $q_\pi(s, \pi'(s)) >= v_\pi(s)$, then the policy $\pi'$ is better than $\pi$

**policy improvement**
$$ \pi'(s) = argmax_a q_\pi(s, a)$$

So far in this section we have considered the special case of deterministic policies.In fact all the ideas of this section extend easily to stochastic policies.

### Policy Iteration

Once a policy $\pi$, has been improved using $v_{\pi}$ to yield a better policy, $\pi'$, we can then compute $v_{\pi'}$ and improve it again to yield an even better $\pi'$. We can thus obtain a sequence of monotonically improving policies and value functions.

Policy iteration (using iterative policy evaluation) -- Page 87

Faster convergence is often achieved by interposing **multiple policy evaluation sweeps between each policy improvement sweep**.

### Value Iteration

The policy evaluation step of policy iteration can be truncated in several ways without losing the convergence guarantees of policy iteration.

Value Iteration can be written as a particularly simple backup operation that **combines** the policy improvement and truncated policy evaluation steps:

$$ v_{k+1} (s) = max_a \sum_{s', r} p(s', r|s, a)[r+\gamma v_{k}(s')]  $$

How the value iteration backup is identical to the policy evaluation backup except that it requires the **maximum** to be taken over all actions.

Value iteration -- Page 90

### Asynchronous Dynamic Programming

A major drawback to the DP methods that we have discussed so far is that they involve operations over the entire state set of the MDP, that is, they require sweeps of the state set.

For Asynchronous DP, the values of some states may be backed up several times before the values of others are backed up once.

To converge correctly, however, an asynchronous algorithm must continue to backup the values of all the states.

To solve a given MDP, we can run an iterative DP algorithm *at the same time that an agent is actually experiencing the MDP.*

### Generalized Policy Iteration

Policy iteration consists of two simultaneous, interacting processes
- policy evaluation: making the value function consistent with the current policy
- policy improvement: making the policy greedy with respect to the current value function

Putting these two computations together, we obtain policy iteration and its variants
- policy iteration, these two processes alternate, each completing before the other begins
- but this is not really necessary
    - value iteration: only a single iteration of policy evaluation is performed in between each policy improvement
    - asynchronous DP methods: the evaluation and improvement processes are interleaved at an even finer grain. In some cases a single state is updated in one process before returning to the other.
- As long as both processes continue to update all states, the ultimate result is typically the same—convergence to the optimal value function and an optimal policy.

We use the term **generalized policy iteration (GPI)** to refer to the general idea of letting policy evaluation and policy improvement processes interact, independent of the granularity and other details of the two processes.

### Effciency of Dynamic Programming

### Summary

Worth reading

## Chapter 5 | Monte Carlo Methods

Monte Carlo methods require only experience -- sample sequences of states, actions, and rewards from actual or simulated interaction with an environment. Unlike the previous chapter, here we do not assume complete knowledge of the environment.

Although a model is required, the model need only generate sample transitions, not the complete probability distributions of all possible transitions that is required for dynamic programming (DP). In surprisingly many cases it is easy to generate experience sampled according to the desired probability distributions, but infeasible to obtain the distributions in explicit form.

### Monte Carlo Prediction

learning the state-value function for a given policy -- simply to average the returns observed after visits to that state. As more returns are observed, the average should converge to the expected value.

1. The first-visit MC method estimates $v_\pi(s)$ as the average of the returns following first visits to $s$
2. the every-visit MC method averages the returns following all visits to $s$.



## Chapter 6 | Temporal-Difference Learning
## Chapter 7 | Multi-step Bootstrapping    
## Chapter 8 | Planning and Learning with Tabular Methods

# Part II -- Approximate Solution Methods
## Chapter 9 | On-policy Prediction with Approximation
## Chapter 10 | On-policy Control with Approximation
## Chapter 11 | Off-policy Methods with Approximation    
## Chapter 12 | Eligibility Traces
## Chapter 13 | Policy Gradient Methods
