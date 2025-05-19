import numpy as np
import tensorflow as tf
from tf_agents.agents.ppo import ppo_agent
from tf_agents.environments import py_environment
from tf_agents.environments import tf_py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts
from tf_agents.networks import actor_distribution_network
from tf_agents.networks import value_network
from tf_agents.drivers import dynamic_step_driver
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.policies import random_tf_policy


class GridWorldEnv(py_environment.PyEnvironment):
    def __init__(self, grid_size=(5, 5), start=(0, 0), goal=(4, 4), obstacles={(1, 1), (2, 2), (3, 3)}, rewards=None):
        # Initialize the environment parameters
        self.grid_size = grid_size  # Size of the grid
        self.start = start  # Starting position of the agent
        self.goal = goal  # Goal position in the grid
        self.obstacles = obstacles  # Positions of obstacles in the grid
        # Set the rewards for each position in the grid
        self.rewards = rewards if rewards is not None else {goal: 1.0, **{state: -1.0 for state in obstacles}}
        # Update rewards for all other positions to 0.0
        self.rewards.update({(i, j): 0.0 for i in range(grid_size[0]) for j in range(grid_size[1]) if
                             (i, j) not in obstacles and (i, j) != goal})
        self._state = self.start  # Initialize the state to the start position
        self._episode_ended = False  # Flag to indicate if the episode has ended

        # Define the action and observation specs
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=3,
            name='action')  # Actions: 0 (up), 1 (right), 2 (down), 3 (left)
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(2,), dtype=np.int32, minimum=0, maximum=max(grid_size) - 1,
            name='observation')  # Observations: 2D position in the grid

    def action_spec(self):
        return self._action_spec  # Return the action specification

    def observation_spec(self):
        return self._observation_spec  # Return the observation specification

    def _reset(self):
        # Reset the environment to the starting state
        self._state = self.start  # Set the state to the start position
        self._episode_ended = False  # Reset the episode ended flag
        return ts.restart(np.array(self._state, dtype=np.int32))  # Return the initial time step

    def _step(self, action):
        if self._episode_ended:
            return self.reset()  # If episode has ended, reset the environment

        # Determine the next state based on the action
        next_state = list(self._state)
        if action == 0:  # Up
            next_state[0] -= 1
        elif action == 1:  # Right
            next_state[1] += 1
        elif action == 2:  # Down
            next_state[0] += 1
        elif action == 3:  # Left
            next_state[1] -= 1

            # Check if the next state is valid (within grid and not an obstacle)
        if (next_state[0] < 0 or next_state[0] >= self.grid_size[0] or next_state[1] < 0 or next_state[1] >=
                self.grid_size[1] or tuple(next_state) in self.obstacles):
            next_state = self._state  # If invalid, revert to current state

        self._state = tuple(next_state)  # Update the state
        reward = self.rewards.get(self._state, -0.1)  # Get the reward for the new state, default to -0.1

        if self._state == self.goal:
            self._episode_ended = True  # Set the episode ended flag
            return ts.termination(np.array(self._state, dtype=np.int32), reward)  # Return the termination time step
        else:
            return ts.transition(np.array(self._state, dtype=np.int32), reward)  # Return the transition time step


def train_ppo_agent(num_iterations=1000):
    # Create the GridWorld environment
    env = GridWorldEnv()
    tf_env = tf_py_environment.TFPyEnvironment(env)  # Convert to a TensorFlow environment

    # Define the actor and value networks
    actor_net = actor_distribution_network.ActorDistributionNetwork(
        tf_env.observation_spec(),  # Observation spec of the environment
        tf_env.action_spec(),  # Action spec of the environment
        fc_layer_params=(100,)  # Fully connected layer parameters (one hidden layer with 100 units)
    )

    value_net = value_network.ValueNetwork(
        tf_env.observation_spec(),  # Observation spec of the environment
        fc_layer_params=(100,)  # Fully connected layer parameters (one hidden layer with 100 units)
    )

    # Define the optimizer
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=1e-3)  # Adam optimizer with learning rate 0.001

    # Create the PPO agent
    ppo_agent_instance = ppo_agent.PPOAgent(
        tf_env.time_step_spec(),  # Time step spec of the environment
        tf_env.action_spec(),  # Action spec of the environment
        optimizer,  # Optimizer for training
        actor_net=actor_net,  # Actor network
        value_net=value_net,  # Value network
        num_epochs=10,  # Number of epochs for training
        train_step_counter=tf.Variable(0)  # Training step counter
    )

    # Initialize the agent
    ppo_agent_instance.initialize()

    # Create the replay buffer
    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        data_spec=ppo_agent_instance.collect_data_spec,  # Data spec for the collected data
        batch_size=tf_env.batch_size,  # Batch size of the environment
        max_length=1000  # Maximum length of the buffer
    )

    # Create a random policy for initial data collection
    random_policy = random_tf_policy.RandomTFPolicy(tf_env.time_step_spec(), tf_env.action_spec())

    # Collect initial data
    initial_collect_driver = dynamic_step_driver.DynamicStepDriver(
        tf_env,  # Environment to collect data from
        random_policy,  # Policy to use for data collection
        observers=[replay_buffer.add_batch],  # Observers to call with each collected step (store in replay buffer)
        num_steps=100  # Number of steps to collect
    )

    initial_collect_driver.run()  # Run the data collection

    # Create a dataset from the replay buffer
    dataset = replay_buffer.as_dataset(
        num_parallel_calls=3,  # Number of parallel calls for data loading
        sample_batch_size=64,  # Batch size for sampling from the buffer
        num_steps=2  # Number of steps in each sample
    ).prefetch(3)  # Prefetch data to improve performance

    iterator = iter(dataset)  # Create an iterator for the dataset

    for _ in range(num_iterations):
        # Get a batch of experiences from the dataset
        experience, unused_info = next(iterator)
        # Train the agent on the batch of experiences
        train_loss = ppo_agent_instance.train(experience)

        # Print the training step and loss
        print(f'Step: {ppo_agent_instance.train_step_counter.numpy()} Loss: {train_loss.loss.numpy()}')


def main():
    # Train the PPO agent
    train_ppo_agent(num_iterations=1000)


if __name__ == "__main__":
    main()