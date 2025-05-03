import numpy as np
import matplotlib.pyplot as plt
import copy

from coin import COIN
from environments import CustomMountainCarEnv
from rl import COINQLearningAgent


def test_coin_q_learning():
    # Set p contexts and scale factors for testing
    scale_factors = np.concatenate([
        0.0*np.ones((1, )),
        1.0*np.ones((1, )),
        0.0*np.ones((1,)),
    ])
    p_context = np.zeros((scale_factors.shape[0], 3))
    p_context[scale_factors == 0.0, 0] = 1.0
    p_context[scale_factors == 1.0, 1] = 1.0

    # Begin training loop
    num_episodes = p_context.shape[0]
    C = p_context.shape[1]-1

    env = CustomMountainCarEnv(amplitude=1.0, force_sf=1.0, render_mode="none")
    # Create the COIN Q-learning agent
    np.random.seed(0)
    agent = COINQLearningAgent(
        env=env,
        max_contexts=C,
        num_position_bins=1,
        num_velocity_bins=1,
        alpha=0.1,
        gamma=0.99,
        epsilon=1.0,
        epsilon_decay=0.999
    )

    rewards = []

    for i in range(0,num_episodes):
        # Create the MountainCar environment with the true amplitude
        env = CustomMountainCarEnv(amplitude=scale_factors[i], render_mode="none")

        # Train the agent in the current context
        # Reset random seed
    
        np.random.seed(i)
        env.action_space.seed(i)

        training_reward = agent.train_step(
            env=env,
            p_context=p_context[i,:],
            max_steps_per_episode=1,
        )
        if i == 2:
            Q1 = agent.Qdat[0].copy()

        rewards.append(training_reward)

        # Print the average training reward every 500 episodes
        if ((i+1)%5000 == 0):
            print(f"Episode {i+1}, Training reward: {np.mean(rewards[-500:])}")
            print(f"Epsilons: {agent.epsdat}")
            print(f"P contexts: {p_context[i,:]}")
    

    # Set p contexts and scale factors for testing
    scale_factors = np.concatenate([
        0.0*np.ones((2, )), 
    ])
    p_context = np.zeros((scale_factors.shape[0], 3))
    p_context[scale_factors == 0.0, 0] = 1.0
    p_context[scale_factors == 1.0, 1] = 1.0

    # Begin training loop
    num_episodes = p_context.shape[0]
    C = p_context.shape[1]-1

    env = CustomMountainCarEnv(amplitude=1.0, force_sf=1.0, render_mode="none")
    # Create the COIN Q-learning agent
    np.random.seed(0)
    agent = COINQLearningAgent(
        env=env,
        max_contexts=C,
        num_position_bins=1,
        num_velocity_bins=1,
        alpha=0.1,
        gamma=0.99,
        epsilon=1.0,
        epsilon_decay=0.999
    )

    rewards = []

    for i in range(0,num_episodes):
        # Create the MountainCar environment with the true amplitude
        env = CustomMountainCarEnv(amplitude=scale_factors[i], render_mode="none")

        # Train the agent in the current context
        # Reset random seed
        np.random.seed(2*i)
        env.action_space.seed(2*i)
        training_reward = agent.train_step(
            env=env,
            p_context=p_context[i,:],
            max_steps_per_episode=1,
        )

        if i == 1:
            Q2 = agent.Qdat[0].copy()

        rewards.append(training_reward)

        # Print the average training reward every 500 episodes
        if ((i+1)%5000 == 0):
            print(f"Episode {i+1}, Training reward: {np.mean(rewards[-500:])}")
            print(f"Epsilons: {agent.epsdat}")
            print(f"P contexts: {p_context[i,:]}")
    

    # Print resulting Q-values
    print(np.all(np.isclose(Q1, Q2, rtol=1e-3)))

def test_function():
    coin_model = COIN(plot_predicted_probabilities=1,
                      particles=2,
                      max_contexts=3)
    # coin_model.perturbations = np.concatenate([
    #     np.zeros((50, )), 
    #     np.ones((125, )), 
    #     -np.ones((15, )), 
    #     np.ones((150, )) * np.nan, 
    # ])
    coin_model.perturbations = np.array([0,1,-1])

    pred_prob = np.zeros((4,2,3))
    pred_prob[0,:,0] = 1
    pred_prob[0:2,:,1] = np.array([[1,0.999],[0,0.001]])
    pred_prob[0:3,:,2] = np.array([[0.4669,0.1146],[0.5329,0.5080],[0.0001,0.3774]])

    S = {}
    S["runs"] = [{"predicted_probabilities":pred_prob}]
    S["runs"][0]["inds_resampled"] = np.array([[0,0,0],[1,1,1]])
    S["runs"][0]["context"] = np.array([[1,2,3],[1,2,3]])
    S["weights"] = 1


    P, S, optimal_assignment, from_unique, context_sequence, C = coin_model.find_optimal_context_labels(S)
    
    # output = coin_model.simulate_coin()
    
    # plt.plot(output["runs"][0]["state_feedback"], "b.", label="state feedback")
    # plt.plot(output["runs"][0]["motor_output"], "r", label="motor output")
    # plt.legend()
    # plt.savefig("figures/temp_test.png")

    output=True
    
    return output


def main():
    retention_values = np.linspace(0.8, 1, 500, endpoint=True)
    drift_values = np.linspace(-0.1, 0.1, 500, endpoint=True)
    state_values = np.linspace(-1.5, 1.5, 500, endpoint=True)
    bias_values = np.linspace(-1.5, 1.5, 500, endpoint=True)
    state_feedback_values = np.linspace(-1.5, 1.5, 500, endpoint=True)

    store = [
        "state_feedback", "motor_output", "responsibilities", "predicted_probabilities", 
    ]

    coin_model = COIN(
        retention_values=retention_values,
        drift_values=drift_values, 
        state_values=state_values, 
        bias_values=bias_values, 
        state_feedback_values=state_feedback_values, 
        store=store, 
        plot_predicted_probabilities = True,
        plot_responsibilities = True,
        runs=10,
    )
    coin_model.perturbations = np.concatenate([
        np.zeros((50, )), 
        np.ones((125, )), 
        -np.ones((15, )), 
        np.ones((150, )) * np.nan, 
    ])

    output = coin_model.simulate_coin()
    
    plt.plot(output["runs"][0]["state_feedback"], "b.", label="state feedback")
    plt.plot(output["runs"][0]["motor_output"], "r", label="motor output")
    plt.legend()
    plt.savefig("figures/temp_test.png")

    output=True
    
    return output


if __name__=="__main__":
    output = test_coin_q_learning()