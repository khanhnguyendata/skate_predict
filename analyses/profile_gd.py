import numpy as np


@profile
def naive_gradient_descent(true_scores, n_iter, n_factors):
    # Initialize baseline score, and scores of all latent factors
    alpha = 0.0005
    random_state = np.random.RandomState(seed=42)
    baseline = random_state.random_sample()    
    event_scores = random_state.random_sample((n_factors, true_scores.shape[1]))
    skater_scores = random_state.random_sample((true_scores.shape[0], n_factors))


    # Step 2: repeat until convergence
    for i in range(n_iter):
        # a. Calculate residual for every event-skater pair
        predicted_scores = skater_scores @ event_scores + baseline
        residuals = predicted_scores - true_scores

        # b. Calculate baseline gradient and update baseline score
        baseline_gradient = np.nansum(residuals)
        baseline = baseline - alpha * baseline_gradient

        # c. For each factor k
        for k in range(n_factors):
            # i. Calculate gradients for each factor
            skater_scores_k = skater_scores[:, [k]]
            event_scores_k = event_scores[[k], :]

            event_gradients_k = np.nansum(residuals * skater_scores_k, axis=0, keepdims=True)
            skater_gradients_k = np.nansum(residuals * event_scores_k, axis=1, keepdims=True)

            # ii. Update scores for each factor
            event_scores[[k], :] = event_scores_k - alpha * event_gradients_k
            skater_scores[:, [k]] = skater_scores_k - alpha * skater_gradients_k

    return baseline, event_scores, skater_scores


@profile
def broadcast_gradient_descent(true_scores, n_iter, n_factors):
    # Initialize baseline score, and scores of all latent factors
    alpha = 0.0005
    random_state = np.random.RandomState(seed=42)
    baseline = random_state.random_sample()
    event_scores = random_state.random_sample((n_factors, true_scores.shape[1]))
    skater_scores = random_state.random_sample((true_scores.shape[0], n_factors))

    # Step 2: repeat until convergence
    for i in range(n_iter):
        # a. Calculate residual for every event-skater pair
        predicted_scores = skater_scores @ event_scores + baseline
        residuals = predicted_scores - true_scores

        # b. Calculate baseline gradient and update baseline score
        baseline_gradient = np.nansum(residuals)
        baseline = baseline - alpha * baseline_gradient

        # c. Calculate gradient and update scores for all factors
        reshaped_residuals = residuals[np.newaxis, :, :]
        reshaped_event_scores = event_scores[:, np.newaxis, :]
        reshaped_skater_scores = skater_scores.T[:, :, np.newaxis]
        event_gradients = np.nansum(residuals * reshaped_skater_scores, axis=1)
        skater_gradients = np.nansum(residuals * reshaped_event_scores, axis=2).T

        event_scores = event_scores - alpha * event_gradients
        skater_scores = skater_scores - alpha * skater_gradients

    return baseline, event_scores, skater_scores


@profile
def matmul_gradient_descent(true_scores, n_iter, n_factors):
    alpha = 0.0005
    
    # Initialize baseline score, and scores of all latent factors
    random_state = np.random.RandomState(seed=42)
    baseline = random_state.random_sample()
    event_scores = random_state.random_sample((n_factors, true_scores.shape[1]))
    skater_scores = random_state.random_sample((true_scores.shape[0], n_factors))

    # Step 2: repeat until convergence
    for i in range(n_iter):
        # a. Calculate residual for every event-skater pair
        predicted_scores = skater_scores @ event_scores + baseline
        residuals = predicted_scores - true_scores

        # b. Calculate baseline gradient and update baseline score
        baseline_gradient = np.nansum(residuals)
        baseline = baseline - alpha * baseline_gradient

        # c. Calculate gradient and update scores for all factors
        residuals = np.nan_to_num(residuals)

        # 2c-i: Calculate gradients for all factors
        event_gradients = skater_scores.T @ residuals
        skater_gradients = residuals @ event_scores.T

        # 2c-ii: Update latent scores for all factors
        event_scores = event_scores - alpha * event_gradients
        skater_scores = skater_scores - alpha * skater_gradients

    return baseline, event_scores, skater_scores


if __name__ == '__main__':
    true_scores = np.load('viz/true_scores.npy')
    naive_gradient_descent(true_scores, 1, 100000)
    broadcast_gradient_descent(true_scores, 1, 100000)
    matmul_gradient_descent(true_scores, 1, 100000)