import numpy as np
import matplotlib.pyplot as plt


class GridWorld:
    def __init__(self, grid_size, start, goal, obstacles, rewards):
        # Initialize the grid world environment
        self.grid_size = grid_size  # Size of the grid (rows, columns)
        self.start = start  # Starting position of the agent
        self.goal = goal  # Goal position in the grid
        self.obstacles = obstacles  # Positions of obstacles in the grid
        self.rewards = rewards  # Rewards associated with different positions in the grid
        self.state = start  # Current state of the agent

    def reset(self):
        # Reset the environment to the starting state
        self.state = self.start
        return self.state

    def is_terminal(self, state):
        # Check if the given state is the terminal state (goal)
        return state == self.goal

    def step(self, action):
        # Take an action and return the next state, reward, and whether the episode is done
        if self.is_terminal(self.state):
            return self.state, 0, True  # If the current state is terminal, return zero reward and done

        next_state = list(self.state)
        if action == 0:  # Up
            next_state[0] -= 1
        elif action == 1:  # Right
            next_state[1] += 1
        elif action == 2:  # Down
            next_state[0] += 1
        elif action == 3:  # Left
            next_state[1] -= 1

            # Check if the next state is valid (within grid and not an obstacle)
        if (next_state[0] < 0 or next_state[0] >= self.grid_size[0]
                or next_state[1] < 0 or next_state[1] >= self.grid_size[1]
                or tuple(next_state) in self.obstacles):
            next_state = self.state  # If invalid, revert to current state

        self.state = tuple(next_state)
        reward = self.rewards.get(self.state, -0.1)  # Get the reward for the new state, default to -0.1
        done = self.is_terminal(self.state)  # Check if the new state is terminal
        return self.state, reward, done


def epsilon_greedy_action(Q, state, epsilon, num_actions=4):
    # Select an action using epsilon-greedy strategy
    if np.random.rand() < epsilon:
        return np.random.choice(num_actions)  # Choose a random action with probability epsilon
    else:
        q_values = [Q(state, action) for action in range(num_actions)]
        return np.argmax(q_values)  # Choose the action with the highest Q-value with probability 1-epsilon


def feature_extractor(state, action, grid_size):
    # Feature representation for a state-action pair
    features = np.zeros((grid_size[0], grid_size[1], 4))
    features[state[0], state[1], action] = 1.0
    return features.flatten()


def run_sarsa_experiment_function_approximation(grid, num_trials, epsilon, alpha, gamma=0.9):
    # Run the SARSA strategy experiment with function approximation
    num_features = grid.grid_size[0] * grid.grid_size[1] * 4
    weights = np.zeros(num_features)  # Initialize weights for the linear function approximator
    cumulative_rewards = np.zeros(num_trials)  # Initialize cumulative rewards

    def Q(state, action):
        features = feature_extractor(state, action, grid.grid_size)
        return np.dot(weights, features)

    for i in range(num_trials):
        state = grid.reset()  # Reset the environment to the start state
        action = epsilon_greedy_action(Q, state, epsilon)  # Select the initial action using epsilon-greedy strategy
        done = False
        while not done:
            next_state, reward, done = grid.step(action)  # Take the action and observe the next state and reward
            next_action = epsilon_greedy_action(Q, next_state,
                                                epsilon)  # Select the next action using epsilon-greedy strategy
            td_target = reward + gamma * Q(next_state, next_action)  # Compute the TD target
            td_error = td_target - Q(state, action)  # Compute the TD error
            features = feature_extractor(state, action, grid.grid_size)
            weights += alpha * td_error * features  # Update the weights
            cumulative_rewards[i] += reward  # Accumulate the reward

            state = next_state  # Move to the next state
            action = next_action  # Move to the next action

            if done:
                break  # End the episode if the goal state is reached

    return cumulative_rewards, weights


def main(grid_size=(4, 4), num_trials=500, epsilon=0.1, alpha=0.5, gamma=0.9):
    start = (0, 0)  # Define the starting position
    goal = (3, 3)  # Define the goal position
    obstacles = {(1, 1), (2, 2)}  # Define obstacle positions
    rewards = {goal: 1.0}  # Define the reward for reaching the goal
    rewards.update({state: -1.0 for state in obstacles})  # Define the penalty for hitting obstacles
    rewards.update({(i, j): 0.0 for i in range(grid_size[0]) for j in range(grid_size[1])
                    if (i, j) not in obstacles and (i, j) != goal})  # Define the default reward for other positions
    grid = GridWorld(grid_size, start, goal, obstacles, rewards)  # Initialize the grid world environment

    cumulative_rewards_sarsa, weights_sarsa = run_sarsa_experiment_function_approximation(grid, num_trials, epsilon, alpha, gamma)

    # Plot cumulative rewards for the SARSA strategy
    plt.figure(figsize=(10, 5))
    plt.plot(cumulative_rewards_sarsa, label='SARSA with Function Approximation')
    plt.xlabel('Trials')
    plt.ylabel('Cumulative Reward')
    plt.title('Cumulative Rewards Over Trials')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main(num_trials=500, epsilon=0.1, alpha=0.5, gamma=0.9)
