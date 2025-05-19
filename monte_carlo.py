import numpy as np
import matplotlib.pyplot as plt


class Ad:
    def __init__(self, true_click_rate):
        self.true_click_rate = true_click_rate
        self.estimated_click_rate = 0.0  # Initialize estimated click rate to 0
        self.num_shown = 0

    def show(self):
        return np.random.rand() < self.true_click_rate


def epsilon_greedy_action(ads, epsilon):
    if np.random.rand() < epsilon:
        return np.random.choice(len(ads))
    else:
        return np.argmax([ad.estimated_click_rate for ad in ads])


def run_experiment(ads, num_trials, epsilon):
    rewards = np.zeros(num_trials)
    estimated_click_rates_over_time = np.zeros((len(ads), num_trials))

    for i in range(num_trials):
        chosen_ad = epsilon_greedy_action(ads, epsilon)
        reward = ads[chosen_ad].show()
        ads[chosen_ad].num_shown += 1
        ads[chosen_ad].estimated_click_rate = ((ads[chosen_ad].num_shown - 1) * ads[
            chosen_ad].estimated_click_rate + reward) / ads[chosen_ad].num_shown
        rewards[i] = reward

        for j, ad in enumerate(ads):
            estimated_click_rates_over_time[j, i] = ad.estimated_click_rate

    cumulative_average = np.cumsum(rewards) / (np.arange(num_trials) + 1)
    return cumulative_average, estimated_click_rates_over_time, [ad.estimated_click_rate for ad in ads]


def run_sarsa_experiment(ads, num_trials, epsilon, alpha, gamma=0.9):
    rewards = np.zeros(num_trials)
    estimated_click_rates_over_time = np.zeros((len(ads), num_trials))

    chosen_ad = epsilon_greedy_action(ads, epsilon)  # Initial action selection

    for i in range(num_trials):
        reward = ads[chosen_ad].show()
        rewards[i] = reward

        next_chosen_ad = epsilon_greedy_action(ads, epsilon)

        # SARSA update
        td_target = reward + gamma * ads[next_chosen_ad].estimated_click_rate
        td_error = td_target - ads[chosen_ad].estimated_click_rate

        ads[chosen_ad].estimated_click_rate += alpha * td_error

        for j, ad in enumerate(ads):
            estimated_click_rates_over_time[j, i] = ad.estimated_click_rate

        chosen_ad = next_chosen_ad

    cumulative_average = np.cumsum(rewards) / (np.arange(num_trials) + 1)
    return cumulative_average, estimated_click_rates_over_time, [ad.estimated_click_rate for ad in ads]


def monte_carlo_simulation(true_click_rates, num_ads, num_trials, epsilon, min_exploration_prob, num_simulations, alpha,
                           gamma=0.9):
    cumulative_averages_greedy = np.zeros((num_simulations, num_trials))
    cumulative_averages_epsilon_greedy = np.zeros((num_simulations, num_trials))
    cumulative_averages_sarsa = np.zeros((num_simulations, num_trials))

    final_estimated_click_rates_greedy = np.zeros((num_simulations, num_ads))
    final_estimated_click_rates_epsilon_greedy = np.zeros((num_simulations, num_ads))
    final_estimated_click_rates_sarsa = np.zeros((num_simulations, num_ads))

    for sim in range(num_simulations):
        print(f"Running simulation {sim + 1}/{num_simulations}...")
        ads_greedy = [Ad(rate) for rate in true_click_rates]
        ads_epsilon_greedy = [Ad(rate) for rate in true_click_rates]
        ads_sarsa = [Ad(rate) for rate in true_click_rates]

        cumulative_average_greedy, _, estimated_click_rates_greedy = run_experiment(ads_greedy, num_trials,
                                                                                    epsilon=min_exploration_prob)
        cumulative_average_epsilon_greedy, _, estimated_click_rates_epsilon_greedy = run_experiment(ads_epsilon_greedy,
                                                                                                    num_trials, epsilon)
        cumulative_average_sarsa, _, estimated_click_rates_sarsa = run_sarsa_experiment(ads_sarsa, num_trials, epsilon,
                                                                                        alpha, gamma)

        cumulative_averages_greedy[sim, :] = cumulative_average_greedy
        cumulative_averages_epsilon_greedy[sim, :] = cumulative_average_epsilon_greedy
        cumulative_averages_sarsa[sim, :] = cumulative_average_sarsa

        final_estimated_click_rates_greedy[sim, :] = estimated_click_rates_greedy
        final_estimated_click_rates_epsilon_greedy[sim, :] = estimated_click_rates_epsilon_greedy
        final_estimated_click_rates_sarsa[sim, :] = estimated_click_rates_sarsa

    mean_cumulative_average_greedy = np.mean(cumulative_averages_greedy, axis=0)
    mean_cumulative_average_epsilon_greedy = np.mean(cumulative_averages_epsilon_greedy, axis=0)
    mean_cumulative_average_sarsa = np.mean(cumulative_averages_sarsa, axis=0)

    mean_final_estimated_click_rates_greedy = np.mean(final_estimated_click_rates_greedy, axis=0)
    mean_final_estimated_click_rates_epsilon_greedy = np.mean(final_estimated_click_rates_epsilon_greedy, axis=0)
    mean_final_estimated_click_rates_sarsa = np.mean(final_estimated_click_rates_sarsa, axis=0)

    return (mean_cumulative_average_greedy, mean_cumulative_average_epsilon_greedy, mean_cumulative_average_sarsa,
            mean_final_estimated_click_rates_greedy, mean_final_estimated_click_rates_epsilon_greedy,
            mean_final_estimated_click_rates_sarsa)


def main(num_ads=3, num_trials=1000, epsilon=0.1, min_exploration_prob=0.01, num_simulations=100, alpha=0.001,
         gamma=0.9):
    true_click_rates = [0.1, 0.2, 0.3]

    (mean_cumulative_average_greedy, mean_cumulative_average_epsilon_greedy, mean_cumulative_average_sarsa,
     mean_final_estimated_click_rates_greedy, mean_final_estimated_click_rates_epsilon_greedy,
     mean_final_estimated_click_rates_sarsa) = monte_carlo_simulation(
        true_click_rates, num_ads, num_trials, epsilon, min_exploration_prob, num_simulations, alpha, gamma)

    print("Starting to plot...")
    plt.figure(figsize=(14, 5))

    # Plot the cumulative average reward for all strategies
    plt.subplot(1, 2, 1)
    plt.plot(mean_cumulative_average_greedy, label='Greedy')
    plt.plot(mean_cumulative_average_epsilon_greedy, label='Epsilon-Greedy')
    plt.plot(mean_cumulative_average_sarsa, label='SARSA')
    plt.xlabel('Trials')
    plt.ylabel('Mean Cumulative Average Reward')
    plt.legend()
    print("Plotted cumulative average reward.")

    plt.tight_layout()
    plt.savefig('plot_output.png')  # Save the plot to a file
    plt.show()  # Display the plot in a non-blocking manner
    plt.pause(1)  # Pause to ensure the plot window is displayed
    print("Finished plotting.")

    # Print the estimated click rates for each ad
    print("Estimated click rates (Greedy):")
    for i, rate in enumerate(mean_final_estimated_click_rates_greedy):
        print(f"Ad {i + 1}: True click rate: {true_click_rates[i]}, Estimated click rate: {rate}")

    print("Estimated click rates (Epsilon-Greedy):")
    for i, rate in enumerate(mean_final_estimated_click_rates_epsilon_greedy):
        print(f"Ad {i + 1}: True click rate: {true_click_rates[i]}, Estimated click rate: {rate}")

    print("Estimated click rates (SARSA):")
    for i, rate in enumerate(mean_final_estimated_click_rates_sarsa):
        print(f"Ad {i + 1}: True click rate: {true_click_rates[i]}, Estimated click rate: {rate}")


if __name__ == "__main__":
    main(epsilon=0.2, num_ads=3, num_trials=1000, min_exploration_prob=0.01, num_simulations=100, alpha=0.005,
         gamma=0.9)