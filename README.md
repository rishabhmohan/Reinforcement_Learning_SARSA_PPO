# Reinforcement Learning  
  
Simple Python scripts to learn reinforcement learning using the epsilon-greedy method and SARSA update.  
  
## Overview  
  
This script implements a Grid World environment and uses the SARSA algorithm to find the optimal policy for navigating the grid. The Grid World is a simple environment where an agent starts at a specified position and attempts to reach a goal position while avoiding obstacles. The agent receives rewards or penalties depending on the positions it visits.  
  
### Epsilon-Greedy Strategy  
  
The epsilon-greedy strategy is a method used in reinforcement learning to balance exploration and exploitation. With probability `epsilon`, the agent selects a random action (exploration) to discover potentially better options. With probability `1 - epsilon`, the agent selects the action that has the highest estimated reward based on the current Q-values (exploitation). This approach helps the agent to explore new actions while still making use of the knowledge it has gained so far.  
  
### SARSA Algorithm  
  
SARSA (State-Action-Reward-State-Action) is an on-policy reinforcement learning algorithm used to learn the optimal policy for an agent navigating an environment. In each step, the agent:  
1. Observes the current state.  
2. Selects an action using an epsilon-greedy strategy.  
3. Takes the action and observes the next state and reward.  
4. Selects the next action using an epsilon-greedy strategy.  
5. Updates the Q-value for the current state-action pair using the observed reward and the estimated value of the next state-action pair.  
  
The Q-value update is given by:  
\[ Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma Q(s', a') - Q(s, a) \right] \]  
where \( s \) is the current state, \( a \) is the current action, \( r \) is the reward, \( s' \) is the next state, \( a' \) is the next action, \( \alpha \) is the learning rate, and \( \gamma \) is the discount factor.  
  
SARSA is called an on-policy algorithm because it updates the Q-values based on the policy currently being followed, which includes the exploration strategy.  
  
## Example  
  
An example setup is provided in the `main` function:  
- **Grid size:** 4x4  
- **Start position:** (0, 0)  
- **Goal position:** (3, 3)  
- **Obstacles:** {(1, 1), (2, 2)}  
- **Rewards:** 1.0 for reaching the goal, -1.0 for hitting obstacles, 0.0 for other positions.  
  
The SARSA experiment is run for 10 trials with `epsilon=0.1`, `alpha=0.5`, and `gamma=0.9`.  
  
### Output  
  
The output of the script includes the cumulative rewards plot, which shows how the agent's performance improves over the trials.  
  
## Dependencies  
  
- `numpy`  
- `matplotlib`  
  
Make sure to install the
