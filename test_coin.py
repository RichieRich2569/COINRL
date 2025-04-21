import numpy as np
import matplotlib.pyplot as plt

from coin import COIN

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
    output = main()