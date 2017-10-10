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

So far in this section we have considered the special case of deterministic policies. In fact all the ideas of this section extend easily to stochastic policies.

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

learning the state-value function **for a given policy** -- simply to average the returns observed after visits to that state. As more returns are observed, the average should converge to the expected value.

1. The first-visit MC method estimates $v_\pi(s)$ as the average of the returns following first visits to $s$
2. the every-visit MC method averages the returns following all visits to $s$.

Whereas the DP diagram includes only one-step transitions, the Monte Carlo diagram goes all the way to the end of the episode.

The advantage of MC DP:
1. the ability to learn from actual experience
2. the ability to learn from simulated experience
3. One can generate many sample episodes starting from the states of interest, averaging returns from only these states ignoring all others.
    - In Monte Carlo methods: the estimates for each state are independent. In DP, however, the estimate for one state builds upon the estimate of any other state.

### Monte Carlo Estimation of Action Values

- With a model, state values alone are suffcient to determine a policy; one simply looks ahead one step and chooses whichever action leads to the best combination of reward and next state
- Without a model, however, state values alone are not suffcient. One must explicitly estimate action values (the values of state–action pairs) rather than state values.

The only complication is that many state–action pairs may never be visited.

### Monte Carlo Control

Monte Carlo ES (Exploring Starts) -- Page 107

## Monte Carlo Control without Exploring Starts

- **On-policy** methods attempt to evaluate or improve the policy that is used to make decisions
- **off-policy** methods evaluate or improve a policy different from that used to generate the data.

$\epsilon$-soft: $\pi(a|s) \ge \frac{\epsilon}{|\mathcal{A}(s)|}$
- Among $\epsilon$-soft policies, $\epsilon$-greedy policies are in some sense those that are closest to greedy.
- Policy iteration works for $\epsilon$-soft
    - Although we only achieve the best policy among the $\epsilon$-soft policies, we have eliminated the assumption of exploring starts.

### Off-policy Prediction via Importance Sampling
All learning control methods face a dilemma: They seek to learn action values conditional on subsequent optimal behavior, but they need to behave non-optimally in order to explore all actions (to find the optimal actions).
- The on-policy approach learns action values not for the optimal policy, but for a near-optimal policy that still explores.
    - Advantage: simpler, are considered first
- Off-policy approach uses two policies, one that is learned about and that becomes the optimal policy, and one that is more exploratory and is used to generate behavior.
    - The policy being learned about is called the **target policy**, and the policy used to generate behavior is called the **behavior policy.**
    - Shortcoming:
        - are often of greater variance, slower to converge.
        - require additional concepts and notation
    - Advantage:
        - more powerful and general
        - They include on-policy methods as the special case in which the target and behavior policies are the same.

Almost all off-policy methods utilize importance sampling, a general technique for estimating expected values under one distribution given samples from another.

importance sampling ratio:
$$
\rho_t^T = \frac{\prod_{k=t}^{T} \pi(A_k|S_k) p(S'_{k+1}|S_k, A_k)}{\prod_{k=t}^{T} \mu(A_k|S_k) p(S'_{k+1}|S_k, A_k)} = \frac{\prod_{k=t}^{T} \pi(A_k|S_k)}{\prod_{k=t}^{T} \mu(A_k|S_k)}
$$

Two kinds of importance sampling methods
1. ordinary importance sampling
$$
V(s) = \frac{ \sum_{t \in \mathcal{T}(s) } \rho_t^{T(t)} G_t  }{ |\mathcal{T}(s)| }
$$
($\mathcal{T}(s)$  only include time steps that were visits to $s$ within their episodes.)
    - unbiased, but its variance is unbounded
2. weighted importance sampling
$$
V(s) = \frac{ \sum_{t \in \mathcal{T}(s) } \rho_t^{T(t)} G_t  }{ \sum_{t \in \mathcal{T}(s) } \rho_t^{T(t)} }
$$
    - biased, but drammatically low variance. strongly prefered in practice

### Incremental Implementation

MC prediction can be implemented imcrementally, just as in Multi-Arm Bandit (just replace reward with return)

$$ G = \gamma G + R_{t+1} $$

$$ C(S_t, A_t) = C(S_t, A_t) + W $$

$$ Q(S_t, A_t) = Q(S_t, A_t) + \frac{W_t}{C(S_t, A_t)} (G-Q(S_t, A_t)) $$

$$ W_{t+1} = W_t * \frac{\pi(A_t|S_t)}{\mu(A_t|S_t)} $$

### Off-Policy Monte Carlo Control

**An advantage of the separation of on-policy and off-policy** is that the target policy may be deterministic (e.g., greedy), while the behavior policy can continue to sample all possible actions.

Off-policy every-visit MC control (returns $\pi \approx \pi^{*}$)

### Summary

Prediction and Control:
- **Prediction**: estimating the state-value function $v_\pi$ for a given policy
- **Control**: control, that is, to approximate optimal poli-
cies

Maintaining suffcient exploration is an issue in Monte Carlo control methods. One approach is to ignore this problem by assuming that episodes begin with state–action pairs randomly selected to cover all possibilities.
- In on-policy methods, the agent commits to always exploring and tries to find the best policy that still explores.
- In off-policy methods, the agent also explores, but learns a deterministic optimal policy that may be unrelated to the policy followed.

Differences between DP and MC:
1. MC operates on sample experience, and thus can be used for direct learning without a model
2. MC does not **bootstrap.** That is, they do not update their value estimates on the basis of other value estimates.

These two differences are not tightly linked, and can be separated.

## Chapter 6 | Temporal-Difference Learning

Temporal-difference means that TD methods **change an earlier estimate based on how it differs from a later estimate**

TD learning is a combination of Monte Carlo ideas and dynamic programming (DP) ideas.
- Like Monte Carlo methods, TD methods can learn directly from raw experience without a model of the environment’s dynamics.
- Like DP, TD methods update estimates based in part on other learned estimates, without waiting for a final outcome (they bootstrap).


prediction and control of TD
1. prediction: The differences in DP, TD, and Monte Carlo methods are primarily differences in their approaches to the prediction problem.
2. control: all use some variation of generalized policy iteration (GPI).

### TD Prediction

- A simple every-visit Monte Carlo method suitable for nonstationary environments is
$$ V(S_t) = V(S_t) + \alpha[G_t - V(S_t)] $$
- **TD(0)**
$$ V(S_t) = V(S_t) + \alpha[R_{t+1} + \gamma V(S_{t+1}) - V(S_t)] $$

> The value of a state under a policy $\pi$  (Chapter 3)
> $$ v_\pi (s) = E_\pi(G_t|s_t=s) = E_{\pi}[r+\gamma v_\pi(s')]  $$

Monte Carlo methods use an estimate of $v_\pi (s) = E_\pi(G_t|s_t=s)$ as a target,
- The Monte Carlo target is an estimate because the expected value in $E_\pi(G_t|s_t=s)$ is not known; a sample return is used in place of the real expected return.

DP methods use an estimate of $v_{\pi} (s) = E_{\pi}[r+\gamma v_\pi(s')]$ as a target.
- The DP target is an estimate not because of the expected values, which are assumed to be completely provided by a model of the environment, but because $v_\pi(S_{t+1})$ is not known and the current estimate, $V(S_{t+1})$, is used instead.

The TD target is an estimate for both reasons:
1. it samples the expected values in $E_{\pi}[r+\gamma v_\pi(s')]$
2. and it uses the current estimate $V$ instead of the true $v_{\pi}$.

Thus, TD methods combine the sampling of Monte Carlo with the bootstrapping of DP.

Sample backups (TD, MC) differ from the full backups (DP) in that they are based on a single sample successor rather than on a complete distribution of all possible successors.

TD error: $\delta_t = R_{t+1} + \gamma V(S_{t+1}) - V(S_t)$
TD error is not actually available until one time step later

$G_t - V(S_t) \\
= R_{t+1} + \gamma V(S_{t+1}) - V(S_t) \\
= R_{t+1} + \gamma V(S_{t+1}) - V(S_t) + V(S_{t+1}) - V(S_{t+1}) \\
= \delta_t + \gamma (G_{t+1} - V(S_{t+1})) \\
= \sum_{k=1}^{T-t+1} \gamma^k\delta_{t+k}$

### Advantages of TD Prediction Methods

TD methods learn their estimates in part on the basis of other estimates. They learn a guess from a guess—they bootstrap.

TD's advantages:
- advantage over DP methods: they do not require a model of the environment, of its reward and next-state probability distributions.
- advantage over Monte Carlo methods: With Monte Carlo methods one must wait until the end of an episode, whereas with TD methods one need wait only one time step. TD methods are naturally implemented in an on-line, fully incremental fashion.

The convergency of TD can be proved, but there is no conclusion on whether TD converges faster than MC or not. In practice, TD methods have usually been found to converge faster than constant-$\alpha$ MC methods on stochastic tasks.

### Optimality of TD(0)

batch updating
- the value function is update only once, by the sum of all the increments. Then all the available experience is processed again with the new value function to produce a new overall increment, and so on, until the value function converges.

Convergence
- Batch Monte Carlo methods always find the estimates that minimize **mean-squared error** on the training set,
- batch TD(0) always finds the estimates that would be exactly correct for the **maximum-likelihood** model of the Markov process.
    - batch TD(0) converges to the certainty-equivalence estimate.
    - If the process is Markov, we expect that ML will produce lower error on future data, even though the MSE is better on the existing data.

### Sarsa: On-Policy TD Control
The quintuple $(S_t, A_t, R_t, S_{t+1}, A_{t+1})$ gives rise to the name *Sarsa* for the algorithm

$S_{t+1}$ is terminal, then $Q(S_{t+1}, A_{t+1})$ is defined as zero

- Choose $A$ from $S$ using policy derived from $Q$ (e.g., $\epsilon$-greedy)
- In each iteration:
    - Take action $A$, observe $R$, $S'$
    - Choose $A'$ from $S'$ using policy derived from $Q$ (e.g., $\epsilon$-greedy)
    $$ Q(S, A) = Q(S, A)  + \alpha[R + \gamma Q(S', A') - Q(S, A)] $$

As in all on-policy methods, we continually estimate $q_\pi$ for the behavior policy $\pi$, and at the same time change $\pi$ toward greediness with respect to $q_\pi$.

### Q-learning: Off-Policy TD Control

In Q-learning, the learned action-value function, Q, directly approximates $q_*$, the optimal action-value function, **independent of the policy being followed.**
(so it is off-policy)

In each iteration:
- Choose $A$ from $S$ using policy derived from $Q$ (e.g., $\epsilon$-greedy)
- Take action $A$, observe $R$, $S'$
$$ Q(S, A) = Q(S, A)  + \alpha[R + \gamma max_a Q(S', a) - Q(S, A)] $$

The $a$ may not be executed by the target policy in the next step, thus this method is off-policy.

### Expected Sarsa

$$ Q(S, A) = Q(S, A)  + \alpha[R + \gamma \sum_a \pi(a|S') Q(S', a) - Q(S, A)] $$

Expected Sarsa is more computational costing, but can eliminates the variance due to the random selection of $A_{t+1}$.

Expected Sarsa > Sarsa > Q-learning

Expected Sarsa to be an on-policy algorithm, but in general we can use a policy different from the target policy $\mu$ to generate behavior, in which case Expected Sarsa becomes an off-policy algorithm.
    - For example, suppose $\mu$ is the greedy policy while behavior is more exploratory; then Expected Sarsa is exactly Q-learning.

### Maximization Bias and Double Learning
Maximization Bias
- One way to view the problem is that it is due to using the same samples (plays) both to determine the maximizing action and to estimate its value.

Solution: double Q-learning
In each iteration:
- Choose $A$ from $S$ using policy derived from $Q$ (e.g., $\epsilon$-greedy)
- Take action $A$, observe $R$, $S'$
- with 0.5 probability:
$$ Q_1(S, A) = Q_1(S, A)  + \alpha[R + \gamma Q_2(S', argmax_a Q_1(S', a)) - Q_1(S, A)] $$
- else:
$$ Q_2(S, A) = Q_2(S, A)  + \alpha[R + \gamma Q_1(S', argmax_a Q_2(S', a)) - Q_2(S, A)] $$

### Games, Afterstates, and Other Special Cases

The reason why afterstates is more effcient for tic-tac-toe is that, a conventional action-value function would map from positions and moves to an estimate of the value, but many position–move pairs produce the same resulting position

### Summary

This chapter: one-step, tabular, model-free TD methods

- In the next two chapters:
    - CH7: multistep forms (a link to Monte Carlo methods)
    - CH8: forms that include a model of the environment (a link to planning and dynamic programming).
- Part 2 of the book
    - various forms of function approximation rather than tables (a link to deep learning and artificial neural networks).


## Chapter 7 | Multi-step Bootstrapping

Multi-step TD can be viewed from different perspectives
1. Monte Carlo methods <--- Multi-step TD ---> one-step TD methods
2. Multi-step methods enable **bootstrapping to occur over longer time intervals**, meanwhile enable **updating the action very fast** to take into account anything that has changed.
3. eligibility traces (Chapter 12)

### n-step TD Prediction

> - Monte Carlo methods: perform a backup for each state based on the entire sequence of observed rewards from that state until the end of the episode.
> - one-step TD methods: its backup is based on just the one next reward, bootstrapping from the value of the state one step later as a proxy for the remaining rewards.

**n-step TD**:

perform a backup based on an intermediate number of rewards: more than one, but less than all of them until termination.

$$ G_t^{(n)} = R_{t+1} + \gamma R_{t+2}  + \gamma^2 R_{t+3} + \ldots + V_{t+n-1}(S_{t+n}) $$

The subscript $t+n-1$ of $V_{t+n-1}(S_{t+n})$ means that $V(S_{t})$ will be updated at the $t+n-1$ timestep.

$$ V_{t+n}(S_{t}) = V_{t+n}(S_{t}) + \alpha [G_t^{(n)} - V_{t+n-1}(S_{t})] $$

In practice, all store and access operations (for $S_t
    $, $A_t$, and $R_t$) can take their index mod n

### n-step Sarsa

The main idea is to simply switch states for actions (state–action pairs) and then use an $\epsilon$-greedy policy.

### n-step Off-policy Learning by Importance Sampling

$$ V_{t+n}(S_{t}) = V_{t+n}(S_{t}) + \alpha \rho_{t}^{t+n} [G_t^{(n)} - V_{t+n-1}(S_{t})] $$

$$
\rho_t^{t+n} = \prod_{k=t}^{min(t+n-1,T-1)} \frac{\pi(A_k|S_k)}{ \mu(A_k|S_k)}
$$

The off-policy version of n-step Expected Sarsa would use the same update as above for Sarsa except that we use $\rho_{t}^{t+n-1}$ instead of $\rho_{t}^{t+n}$. This is because in Expected Sarsa **all possible actions are taken into account in the last state**; the one actually taken has no effect and does not have to be corrected for.

The importance sampling has high variance -- after all, the data is less relevant to what you are trying to learn.


### Off-policy Learning Without Importance Sampling: The n-step Tree Backup Algorithm

Is off-policy learning possible without importance sampling?
- The Q-learning and Expected Sarsa methods from Chapter 6 prove that off-policy learning can be done in the one-step case
- the tree-backup algorithm is the corresponding n-step method
    - See the diagram in Page 160

$$ V(S_{t+1}) = \sum_a \pi(a|S_t) Q_{t-1}(S_t, A_t)$$

TD error: $\delta_t = R_{t+1} + \gamma V(S_{t+1}) - Q(S_t, A_t)$

$$ G_t^{(n)} = Q_{t-1}(S_t, A_t) \sum_{k=t}^{min(t+n-1, T-1)} \delta_k \prod_{i=k+1}^{k} \gamma \pi(A_i|S_i)$$

$$ V_{t+n}(S_{t}) = V_{t+n}(S_{t}) + \alpha [G_t^{(n)} - V_{t+n-1}(S_{t})] $$

### A Unifying Algorithm: n-step Q($\sigma$)
1. n-step Sarsa has all sampled transitions

2. the tree-backup algorithm has all state-to-action transitions fully branched without sampling
3. n-step Expected Sarsa backup has all sam- ple transitions except for the last state-to-action ones, which are fully branched with an expected value.

Their diagrams is available in Page 162.

One idea for unification is that,
- one might decide on a step-by-step basis whether one wanted to take the action as a sample, as in Sarsa,
- or consider the expectation over all actions instead, as in the tree backup.

Let  $\sigma_t \in [0, 1]$ denote the degree of sampling on step t. The random variable $\sigma_t might be set as a function of the state, action, or state–action pair at time t.
- $\sigma_t= 1$: full sampling
- $\sigma_t= 0$: a pure expectation with no sampling.

Details are available in the book.

### Summary

Drawbacks:
1. delay
2. computation and memory

Eventually, in Chapter 12, we will see how multi-step TD methods can be implemented with minimal memory and computational complexity using eligibility traces, but there will always be some additional computation beyond one-step methods.

Some comments on multi-step off-policy TD
- importance sampling is conceptually simple but can be of high variance.
- Tree backups: the natural extension of Q-learning to the multi-step case with stochastic target policies. It involves no importance sampling but, again if the target and behavior policies are substantially different, the bootstrapping may span only a few steps even if n is large.

## Chapter 8 | Planning and Learning with Tabular Methods

In Chapter 7, we unify MC and TD by n-step methods.

In this Chapter, we will unify planning and learning methods
- planning methods: methods that **require a model** of the environment, such as dynamic programming and heuristic search
- learning methods: methods that can be used **without a model**, such as Monte Carlo and temporal-di↵erence methods.

Similarity of planning and learning
- The heart of both learning and planning methods is the estimation of value functions by backup operations.


### Models and Planning

$p(s', r|s, a)$
- a state and an action --> model --> next state and next reward (with probability)

**distribution and sample models**
1. distribution models: models produce a description of all possibilities and their probabilities
2. sample models: models produce just one of the possibilities, sampled according to the probabilities

Distribution models are stronger than sample models in that they can always be used to produce samples, but in many applications it is much easier to obtain sample models.

**Simulation:**
- Given a starting state and a policy, a sample model could produce an entire episode, and a distribution model could **generate all possible episodes and their probabilities**.

**Definition of Planning**
model --planning--> policy

**two distinct approaches to planning**
1. state-space planning
2. plan-space planning
    - e.g., evolutionary methods and partial-order planning
    - difficult to apply effciently to the stochastic optimal control problems that are the focus in RL, and we do not consider them further

The unified view of state-space planning methods
- model --> stimulated experience --backups-->values --> policy

Random-sample one-step tabular Q-planning: Page 169

### Dyna: Integrating Planning, Acting, and Learning

Within a planning agent, there are at least two roles for real experience:
1. model-learning -- is involved in planning
    - improve the model (to make it more accurately match the real environment
    - Indirect methods often **make fuller use of a limited amount of experience** and thus achieve a better policy with fewer environmental interactions.
2. direct reinforcement learning
    - directly improve the value function and policy using the kinds of reinforcement learning methods we have discussed in previous chapters.
    - direct methods are much simpler and are **not affected by biases in the design of the model**

Tabular Dyna-Q: Page 172

Direct reinforcement learning, model-learning, and planning are implemented by steps (d), (e), and (f), respectively. If (e) and (f) were omitted, the remaining algorithm would be one-step tabular Q-learning.

### When the Model Is Wrong

The environment may change.

Greater diffculties arise when the environment changes to become better than it was before, and yet the formerly correct policy does not reveal the improvement. In these cases the modeling error may not be detected for a long time.
- Solution: This agent keeps track for each state–action pair of how many time steps have elapsed since the pair was last tried in a real interaction with the environment. The more time that has elapsed, the greater (we might presume) the chance that the dynamics of this pair has changed and that the model of it is incorrect. To encourage behavior that tests long-untried actions, a special “bonus reward” is given on simulated experiences involving these actions.

### Prioritized Sweeping

In the Dyna agents presented in the preceding sections, simulated transitions are started in state–action pairs selected uniformly at random from all previously ex- perienced pairs. But a uniform selection is usually not the best; planning can be much more e cient if simulated transitions and backups are focused on particular state–action pairs.
- the only useful one-step backups are those of actions that lead directly into the one state whose value has been changed. If the values of these actions are updated, then the values of the predecessor states may change in turn.
- A queue is maintained of every state–action pair whose estimated value would change nontrivially if backed up, prioritized by the size of the change.

Prioritized sweeping for a deterministic environment: Page 178

### Planning as Part of Action Selection

There tends to be two ways of thinking about planning. T
1. planning as the gradual improvement of a policy or value function that is **good in all states** generally rather than focused on any particular state (e.g., dynamic programming and Dyna)
2. Planning as Part of Action Selection
- begun and completed after encountering each new state S_t, as a computation whose output is not really a policy, but rather a single decision, the action A_t; on the next step the planning begins anew with S_{t+1} to produce A_{t+1}, and so on.
- It is just that now the values and policy are specific to the current state and its choices, so much so that they are typically **discarded after being used to select the current action.**
- Planning within action selection is most useful in applications in which fast re- sponses are not required.


### Heuristic Search

planning-as-part-of-action-selection -- heuristic search.
- In heuristic search, for each state encountered, a large tree of possible continuations is considered. The approximate value function is applied to the leaf nodes and then backed up toward the current state at the root. The backing up within the search tree is just the same as in the full backups with maxes (those for v^* and q^* ) discussed throughout this book.
- The backing up stops at the state–action nodes for the current state.
- Once the backed-up values of these nodes are computed, the best of them is chosen as the current action, and then all backed-up values are discarded.

This great focusing of memory and computational resources on the current decision is presumably the reason why heuristic search can be so e↵ective.

### Monte Carlo Tree Search

Monte Carlo Tree Search (MCTS) is one of the simplest examples of planning as part of the policy.
- MCTS typically involves no approximate value functions or policies that are retained from one time step to the next; these are computed on each step and then discarded.
- During a step, many simulated trajectories are generated started from the current state and running all the way to a terminal state (or until discounting makes any further reward negligible as a contribution to the return).
    - default policy (equi-probable random policy) --> cheap to compute --> many simulated trajectories can be generated in a short period of time
    - the value of a state–action pair is estimated as the average of the (simulated) returns from that pair.

Each iteration of MCTS proceeds in four stages (Figure 8.11, Page 185):
1. selection
2. expansian
3. simulation
4. backpropogation


# Reinforcement Learning
# Part II -- Approximate Solution Methods
tabular methods --> approximation
- motivation
    1. ininite state space
    2. generalization

## Chapter 9 | On-policy Prediction with Approximation

### Value-function Approximation

backups: as updates to an estimated value function that **shift its value** at particular states **toward a “backed-up value” for that state.**
- Monte Carlo backup: $S_t \mapsto G_t$
- TD(0) backup: $S_t \mapsto R_{t+1} + \gamma \hat{v}(S_{t+1}, \theta_{t})$
- n-step TD backup: $S_t \mapsto G_t^n$
- DP: $S_t \mapsto E(R_{t+1} + \gamma \hat{v}(S_{t+1}, \theta_{t})|S_t=s)$

the table entry for s’s estimated value has simply been shifted a fraction of the way toward g, and the estimated values of all other states were left unchanged.

However, not all function approximation methods are equally well suited for use in reinforcement learning.
- Online training
- capacity to handle nonstationary target functions (target functions that change over time).

### The Prediction Objective (MSVE)

For approximation methods, an update at one state affects many others, and it is not possible to get all states exactly correct, which is different from tabular methods.

Mean Squared Value Error (MSVE):
$$ MSVE(\theta) = \sum_{s\in\mathcal{S} } d(s) [v_\pi(s)-\hat{v}(s, \theta)]^2 $$

d(s) is the fraction of time spent in s under the target policy $\pi$ (on-policy distribution) -- computation details available on Page 193 of the book.

Shortcoming: It is not completely clear that the MSVE is the right performance objective for RL -- after all, our ultimate purpose is to use value function to **find a better policy**.

### Stochastic-gradient and Semi-gradient Methods

a natural learning algorithm for this case is n-step semi-gradient TD, which includes gradient MC and semi-gradient TD(0) algorithms as the special cases when $n=\infty$ and $n=1$

Stochastic gradient- descent (SGD)

$\theta_{t+1} = \theta_{t+1} - \frac{1}{2} \alpha \nabla [v_\pi(s)-\hat{v}(s, \theta)]^2$

$=\theta_{t+1} + (v_\pi(s)-\hat{v}(s, \theta)) \nabla \hat{v}(s, \theta)$

**Gradient Monte Carlo Algorithm for Approximating $\hat{v} \approx v^\pi$**
- generate infinite number of episodes
- for each episode, traverse each step of episode:
    - $=\theta_{t+1} + (G_t-\hat{v}(s, \theta)) \nabla \hat{v}(s, \theta)$

Semi-gradient TD methods are not true gradient methods. In such bootstrapping methods (including DP), the weight vector appears in the update target, yet this is not taken into account in computing the gradient—thus they are semi-gradient methods.

Nevertheless, good results can be obtained for semi-gradient methods in the special case of linear function approximation

**Semi-gradient TD(0) for estimating $\hat{v} \approx v^\pi$ -- (bootstrapping)**
- Repeat (for each episode):
    - init $S$
    - Repeat (for each step of episode):
        - choose $A ~ \pi(\cdot|s)$
        - Take action A, observe $R, S'$
        - $=\theta_{t+1} + (R_t + \gamma \hat{v}(s', \theta) -\hat{v}(s, \theta)) \nabla \hat{v}(s, \theta)$
        - $S \leftarrow S'$

### Linear Methods

Convergence.

#### 1. Polynomials
this case generalizes poorly in the online learning setting typically considered in reinforcement learning.

#### 2. Fourier Basis
#### 3. Coarse Coding
#### 4. Tile Coding
Tile coding is a form of coarse coding for multi-dimensional continuous spaces that is flexible and computationally efficient.

It may be the most practical feature representation for modern sequential digital computers.

I think it is a kind of feature discretization, which is very useful in machine learning engineering.

#### 5. Radial Basis Functions
Radial basis functions are useful for one- or two-dimensional tasks in which a smoothly varying response is important.

### Nonlinear Function Approximation: Artificial Neural Networks

### Least Square TD

LSTD is the most data-e cient linear TD prediction method, but requires computation proportional to the square of the number of weights, whereas all the other methods are of complexity linear in the number of weights.

### Summary
Linear semi-gradient n-step TD is guaranteed to converge under standard condi- tions, for all n, to a MSVE that is within a bound of the optimal error. This bound is always tighter for higher n and approaches zero as $n \rightarrow \infty$. However, in practice that choice results in very slow learning, and some degree of bootstrapping ($1 < n < \infty$) is usually preferrable.

## Chapter 10 | On-policy Control with Approximation

### Episodic Semi-gradient Control

Episodic Semi-gradient Sarsa for Control is almost the same as tabular Sarsa. The only difference is that $q(S, A)$ is approximated by $\hat{q}(S,A,\theta)$

### n-step Semi-gradient Sarsa

### Average Reward: A New Problem Setting for Continuing Tasks

continuing problems: no discounting

$$\eta(\pi) = \sum_s d_\pi(s) \sum_a \pi(a|s) \sum_{s', r}p(s', r|s, a) r $$

ergodicity: $\eta(pi)$ (it is actually $lim_{T\rightarrow \infty }Pr(S_t  = s|A_{0:t-1}, S_0 \sim \pi$) exist and to be independent of $S_0$
- temporal expectation = spatial expectation
- It means that where the MDP starts or any early decision made by the agent can have only a temporary e↵ect; in the long run your expectation of being in a state depends only on the policy and the MDP transition probabilities.
- Ergodicity is suffcient to guarantee the existence of the limits in the equations above.

In the average-reward setting, returns are defined in terms of differences between rewards and the average reward:
$$G_t = R_{t+1} - \eta(\pi) +  R_{t} - \eta(\pi) + \ldots$$

Differential value functions also have Bellman equations

### Deprecating the Discounted Setting

For continue problems, discounting is not useful. The proof can be done via the summation of geometric progression (GP)

### n-step Differential Semi-gradient Sarsa

The n-step TD error has the same form as before. The only difference is how $G_t$ is calculated.

## Chapter 11 | Off-policy Methods with Approximation  

The challenge of off-policy learning can be divided into two parts
1. one which arises in the tabular case
    - has to do with the target of the learning update
    - importance sampling and rejection sampling
2. the other only in the approximate
    - has to do with the distribution of the updates
    - the distribution of updates in the off-policy case is not according to the on-policy distribution
    - Two approaches
        1. to use importance sampling methods again, to warp the update distribution back to the on-policy distribution, so that semi- gradient methods are guaranteed to converge (in the linear case).
        2. to develop true gradient methods that do not rely on any special distribution for stability.

This is a cutting-edge research area, and it is not clear which of these approaches is most effective in practice.

### Semi-gradient Methods

It is just like the corresponding on-policy algorithm except for the addition of $\rho_t$

$$\theta_{t+1} = \theta_{t} + \alpha \rho_t \delta_t \nabla \hat{v} (S_t, \theta) $$

### Baird’s Counterexample

Baird’s counterexample is one of the most straightforward cases where semi-gradient and other simple algorithms are unstable and diverge.

### The Deadly Triad

The danger of instability and divergence arises whenever we combine three things （any two of these three is fine, though）:
1. training on a distribution of transitions other than that naturally generated by the process whose expectation is being estimated (e.g., off-policy learning)
2. scalable function approximation (e.g., linear semi-gradient)
3. bootstrapping (e.g, DP, TD learning)

Note that
1. the danger is not due to control or GPI; it arises prediction as well.
2. It is also not due to learning, as it occurs in planning methods such as dynamic programming.

## Chapter 12 | Eligibility Traces

### The $\lambda$-return

The TD($\lambda$) algorithm can be understood as one particular way of averaging n-step backups. This average contains all the n-step backups, each weighted proportional to $\lambda^{n-1}$

$$ G^\lambda_t = (1-\lambda) \sum_{n=1}^\infty \lambda^{n-1} G^{n}_t$$

we can separate these post-termination terms from the main sum, yielding

$$ G^\lambda_t = (1-\lambda) \sum_{n=1}^{T-t-1}\lambda^{n-1} G^{n}_t + \lambda^{T-t-1} G_t$$

1. $\lambda=1$ -- Monte Carlo algorithm
2. $\lambda=0$ -- $G_t^(1)$, a one-step TD method

off-line $\lambda$-return algorithm.

### TD($\lambda$)

TD($\lambda$) improves over the off-line $\lambda$-return algorithm in three ways
1. it updates the weight vector on every step of an episode rather than only at the end, and thus its estimates may be better sooner.
2. its computations are equally distributed in time rather that all at the end of the episode.
3. it can be applied to continuing problems rather than just episodic problems.

Semi-gradient TD($\lambda$) for value prediction \
In each iteration
- The eligibility trace: $e_t = \nabla \hat{v}(S_t, \theta_t) + \gamma\lambda e_{t-1}$
    - the trace is said to indicate the eligibility of each component of the weight vector for undergoing learning changes should a reinforcing event occur.
- TD error: $\delta_t = R_{t+1} + \gamma V(S_{t+1}) - V(S_t)$ (the same as before)
- $\theta_{t+1} = \theta_t + \alpha\delta_t e_t$

**TD(0)** \
If $\lambda=0$, the trace at t is exactly the value gradient corresponding to S_t. Thus the TD($\lambda$) update reduces to the one-step semi-gradient TD update treated in Chapter 9 (and, in the tabular case, to the simple TD rule).

**TD(1)**
- TD(1) a way of implementing Monte Carlo algorithms that is more general than those presented earlier and that significantly increases their range of applicability. Whereas the earlier Monte Carlo methods were limited to episodic tasks, TD(1) can be applied to discounted continuing tasks as well.
- Moreover, TD(1) can be performed incrementally and on-line, while Monte Carlo methods can learn nothing from an episode until it is over.

The primary weakness of the off-line $\lambda$-return algorithm
- it is off-line: it learns nothing until the episode is finished.

Disadvantage of TD($\lambda$): \
- TD($\lambda$) is more sentitive to $\alpha$

### An On-line Forward View

h-truncated $\lambda$-return

$$ G^{\lambda|h}_t = (1-\lambda) \sum_{n=1}^{h-t-1}\lambda^{n-1} G^{n}_t + \lambda^{h-t-1} G_t^{h-t}$$

The online $\lambda$-return algorithm

 it is very complex

### True Online TD($\lambda$)

### Dutch Traces in Monte Carlo Learning

## Chapter 13 | Policy Gradient Methods

In this chapter we consider methods that instead learn a parameterized policy that can select actions without consulting a value function.

$$\theta_{t+1} = \theta_{t} + \alpha \hat{\nabla \eta(\theta_t)} $$

All methods that follow this general schema we call policy gradient methods.
- $\eta(\theta_t)$: performance measure
    - episodic case: $\eta(\theta_t) = v_{\pi_{\theta}}(s_0)$ (the value of the start state under the parameterized policy)
    - continuing case: $\eta(\theta_t) = r(\theta)$ (average reward rate)
- A value function may still be used to learn the policy weights, but is not required. Methods that learn approximations to both policy and value functions are often called actor–critic methods
    - ‘actor’ is a reference to the learned policy
    - ‘critic’ refers to the learned value function, usually a state- value function.

### Policy Approximation and its Advantages

the policy can be parameterized in any way, as long as $\pi(a|s \theta)$ is differentiable with respect to its weights ($\equiv$ that $\nabla_\theta \pi(a|s \theta$ exists and is always finite$)
 - In practice, to ensure exploration we generally require that the policy never becomes deterministic $0 < \pi(a|s \theta) < 1$

**advantages of policy-based methods over action-value methods**
1. selecting actions according to the softmax in action preferences can approach determinism, whereas with $\epsilon$-greedy action selection over action values, there is always an $\epsilon$ probability of selecting a random action.
2. the policy may be a simpler function to approximate for some problems
3. In problems with significant function approximation, the best approximate policy may be stochastic. Action-value methods have no natural way of finding stochastic optimal policies, whereas policy approximating methods can.
4. the choice of policy parameterization is sometimes a good way of injecting prior knowledge about the desired form of the policy into the rein- forcement learning system.

### The Policy Gradient Theorem

policy gradient theorem:

$$\nabla \eta(pi) = \sum_s d_\pi(s) \sum_a q_\pi(s, a) \nabla_\theta \pi(a|s, \theta)$$

### REINFORCE: Monte Carlo Policy Gradient

repeat forever
- generate an episode, following the latest $\pi_\theta$
    - For each step of the episode $t = 1, \ldots, T-1$
        - $\theta_{t+1} = \theta_{t} + \alpha \gamma^t G_t    \frac{\nabla_\theta \pi(A_t|S_t, \theta)}{\pi(A_t|S_t, \theta)}$

- The update increases the weight vector in this direction proportional to the return (numertor)
    - because it causes the weights to move most in the directions that favor actions that yield the highest return
- The update is also inversely proportional to the action probability. (denominator)
    - because otherwise actions that are selected frequently are at an advantage (the updates will be more often in their direction) and might win out even if they do not yield the highest return.

- Advantage: As a stochastic gradient method, REINFORCE has good theoretical convergence properties.
- Disadvantage: as a Monte Carlo method REINFORCE may be of high variance and thus slow to learn.

### REINFORCE with Baseline

$$\nabla \eta(pi) = \sum_s d_\pi(s) \sum_a (q_\pi(s, a) - b(s)) \nabla_\theta \pi(a|s, \theta)$$

The baseline can be any function, even a random variable, as long as it does not vary with a.

the equation remains true, because $b(s) \sum_a  \nabla_\theta \pi(a|s, \theta) = 0$. The baseline leaves the expected value of the update un- changed, but it can have a large e↵ect on its variance.

The idea of baseline is generalized from gradient bandit method. In the bandit algorithms the baseline was just a number (the average of the rewards seen so far), but for MDPs the baseline should vary with state.

One natural choice for the baseline is an estimate of the state value $\hat{v}(S_t, w)$. This can be calculated with Gradient Monte Carlo Prediction Algorithm, which is introduced in Chapter 9.

### Actor-Critic Methods

REINFORCE-with-baseline is not an actor-critic method because its state-value function is used only as a baseline, not as a critic.
- Reason: it is not used for **bootstrapping** (updating a state from the estimated values of subsequent states), but only as a baseline for the state being updated.
    - the bias introduced through bootstrapping and reliance on the state representation is often on balance beneficial because it reduces variance and accelerates learning.
- REINFORCE with baseline is unbiased and will converge asymptotically to a local minimum, but like all Monte Carlo methods it tends to be slow to learn (high variance) and inconvenient to implement online or for continuing problems.

**One-step Actor-Critic (episodic)**
One-step actor-critic methods replace the full return of REINFORCE with the one-step return.
- fully online and incremental
- Extended to
    - multi-step methods: $G_t^{(1)}$ --> $G_t^{(n)}$
    -  **Actor-Critic with Eligibility Traces (episodic)**: $G_t^(1)$ --> $G_t^\lambda$

### Policy Gradient for Continuing Problems (Average Reward Rate)

With some alternate definitions, the policy gradient theorem as given for the episodic case remains true for the continuing case.

### Policy Parameterization for Continuous Actions

Instead of computing learned probabilities for each of the many actions, we instead compute learned the statistics of the probability distribution. For example, the action set might be the real numbers, with actions chosen from a normal (Gaussian) distribution.

The End
