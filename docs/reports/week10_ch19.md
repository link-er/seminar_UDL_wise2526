
# Report – Week 10: (Chapter 19) Reinforcement Learning

**Presenter:** Marc Gläser

**Date:** 29.01.2026
[![Youtube Video](https://github.com/user-attachments/assets/a79ac59f-1687-4c16-8bf0-000debb885c5)](https://www.youtube.com/watch?v=jSoTA0l9yLU)


## Summary

### Overview

**Reinforcement Learning (RL):** A framework in which an **agent** learns a policy $\pi$ to map states to actions in an **environment**, aiming to maximize the cumulative **reward** over time.

**Key Challenges:**
- **Temporal Credit Assignment:** Determining which past action caused a current reward is hard.
- **Exploration vs. Exploitation:** Balancing trying new actions (gathering data) vs. using known best actions (maximizing reward).
    
### 1. Markov Decision Processes (MDPs)

Mathematical formalization of the RL problem:
**Definitions:**
- **State ($s \in \mathcal{S}$):** Complete description of the world configuration. (Everything is a state)
- **Action ($a \in \mathcal{A}$):** discrete or continuous choices available to the agent. (For example left, right, up, down)
- **Transition ($P(s'|s, a)$):** Probability of moving to state $s'$ given action $a$ in state $s$. (Can be 1 but normally isn't)
- **Reward ($r(s, a)$):** Immediate scalar feedback signal. 
- **Discount Factor ($\gamma \in [0, 1]$):** Weights future rewards. Ensures convergence.

**Goal:** Maximize the **Expected Return** ($G_t$):

$$G_t = \sum_{k=0}^{\infty} \gamma^k r_{t+k+1}$$
### 2. Value Functions & Bellman Equations

To solve the credit assignment problem, we estimate the long-term value of states:

**Value Definitions:**
- **State-Value Function $v_{\pi}(s)$:** Expected return starting from $s$ following policy $\pi$.
- **Action-Value Function $q_{\pi}(s, a)$:** Expected return taking action $a$ in $s$, then following $\pi$.

**The Bellman Expectation Equation:**

Decomposes value into _immediate reward_ + _discounted value of successor state_. 

$$v_{\pi}(s) = \sum_a \pi(a|s) \sum_{s'} P(s'|s,a) [r + \gamma v_{\pi}(s')]$$
### 3. Learning Paradigms

Different options for computing these values are:

| **Method**                   | **Description**                                                                                         | **Characteristics**                                                        |
| ---------------------------- | ------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------- |
| **Dynamic Programming (DP)** | Solves Bellman equations iteratively using the true Transition model $P$. (which we normally dont have) | **Model-Based.** Requires full knowledge of environment.                   |
| **Monte Carlo (MC)**         | Estimates value by averaging returns $G_t$ from complete episodes.                                      | **Model-Free.** High variance, unbiased. Requires termination.             |
| **Temporal Difference (TD)** | Updates estimates based on other estimates (_bootstrapping_) after a single step.                       | **Model-Free.** Lower variance, biased. Update target: $r + \gamma v(s')$. |

### 4. Value-Based Methods 

**Strategy:** Learn the optimal action-value function $q^*(s,a)$ and define policy as $\pi(s) = \text{argmax}_a q(s,a)$.

**Tabular Q-Learning (TD Control):**
Updates $q$-values toward the _optimal_ future value (off-policy), regardless of the actual action taken next.

$$q(s,a) \leftarrow q(s,a) + \alpha [r + \gamma \max_{a'} q(s', a') - q(s,a)]$$

**Deep Q-Learning (DQN):**

Approximates the table with a Neural Network $f[s, \phi] \approx q^*(s,a)$.
- **Loss Function:** $\mathcal{L}(\phi) = (y - q(s, a, \phi))^2$
- **Target ($y$):** $r + \gamma \max_{a'} q(s', a', \phi^-)$
    

**Stability Modifications:**

1. **Experience Replay:** Buffer $\mathcal{D}$ stores transitions $(s,a,r,s')$. Random sampling breaks temporal correlation.
2. **Target Networks:** Uses frozen parameters $\phi^-$ to compute targets, preventing "chasing your own tail."
    
3. **Double DQN (DDQN):**
    - _Problem:_ Max operator in Q-learning systematically overestimates values.
    - _Solution:_ Decouple selection (online net $\phi$) and evaluation (target net $\phi^-$).
    - _Target:_ $y = r + \gamma q(s', \text{argmax}_{a'} q(s', a', \phi), \phi^-)$.
        

### 5. Policy-Based Methods (Policy Gradients)

**Strategy:** Parameterize the policy $\pi(a|s, \theta)$ directly and optimize expected return $J(\theta)$ via Gradient Ascent.

**The Policy Gradient Theorem:**

The gradient update is calculated by using the Formular:

$$
\theta \leftarrow \theta+\alpha \cdot \frac{1}{I} \sum_{i=1}^I \sum_{t=1}^T \frac{\partial \log \left[\pi\left[a_{i t} \mid \mathrm{s}_{i t}, \theta\right]\right]}{\partial \theta} \sum_{k=t}^T r_{i, k+1} .
$$


**REINFORCE (Monte Carlo PG):**

Uses the full actual return $G_t$ from the episode.
- _Issue:_ High variance because $G_t$ depends on long stochastic trajectories.

**Actor-Critic (TD PG):**
Reduces variance by replacing $G_t$ with a lower-variance estimate.
- **Actor ($\theta$):** Updates policy using the **Advantage**.
- **Critic ($\phi$):** Learns value function $v(s, \phi)$ to compute the baseline.
- **Advantage Function:** Quantifies how much better an action was than expected.
    $$A(s_t, a_t) = \underbrace{r_t + \gamma v(s_{t+1}, \phi)}_{\text{TD Target}} - \underbrace{v(s_t, \phi)}_{\text{Baseline}}$$

## Discussion Notes
