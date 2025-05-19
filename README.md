# Reinforcement Learning  
  
Python scripts to learn reinforcement learning with easy examples using the epsilon-greedy method and Q-values updates with SARSA, Q-learning Monte-Carlo, function approximation etc. 
  
## Overview  
  
This script implements a Grid World environment and uses various algorithms like SARSA, Q-learning and MC (in same SARSA script) to find the optimal policy for navigating the grid. The Grid World is a simple environment where an agent starts at a specified position and attempts to reach a goal position while avoiding obstacles. The agent receives rewards or penalties depending on the positions it visits.  
  
### Epsilon-Greedy Strategy  
  
The epsilon-greedy strategy is a method used in reinforcement learning to balance exploration and exploitation. With probability `epsilon`, the agent selects a random action (exploration) to discover potentially better options. With probability `1 - epsilon`, the agent selects the action that has the highest estimated reward based on the current Q-values (exploitation). This approach helps the agent to explore new actions while still making use of the knowledge it has gained so far.  
  
### SARSA Algorithm  
  
SARSA (State-Action-Reward-State-Action) is an on-policy reinforcement learning algorithm used to learn the optimal policy for an agent navigating an environment. In each step, the agent:  
1. Observes the current state.  
2. Selects an action using an epsilon-greedy strategy.  
3. Takes the action and observes the next state and reward.  
4. Selects the next action using an epsilon-greedy strategy.  
5. Updates the Q-value for the current state-action pair using the observed reward and the estimated value of the next state-action pair.  

### SARSA vs Q-learning
You can update Q values with SARSA update or Q-learning update:
1. SARSA: On-policy algorithm. Updates Q-values based on the action actually taken by the current policy.
2. Q-Learning: Off-policy algorithm. Updates Q-values based on the action that maximizes the future reward, regardless of the current policy.

### Monte-Carlo vs SARSA
SARSA updates Q-values using the TD (temporal difference) error based on the current state, action, reward, next state, and next action, relying on bootstrapping. 
In contrast, Monte Carlo methods update Q-values based on the average of the returns observed from complete episodes without relying on intermediate estimates.

### SARSA with Function_approximation
To add function approximation, we can use a linear function approximator with a set of features. We'll use a simple feature representation for each state-action pair and update the weights of the linear model instead of using a tabular Q-learning approach.

### PPO Algorithm
 
Proximal Policy Optimization (PPO) is a policy gradient method for reinforcement learning that alternates between sampling data through interaction with the environment and optimizing a "surrogate" objective function using stochastic gradient ascent. 
In each step, the agent:

1. Observes the current state.
2. Selects an action based on its current policy.
3. Takes the action and observes the next state and reward.
4. Updates the policy using the observed rewards to maximize the expected reward.

PPO uses clipped probability ratios to limit the change in policy at each update step, which helps to ensure stable and reliable training.


