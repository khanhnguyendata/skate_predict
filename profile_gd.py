import numpy as np


@profile
def naive_gradient_descent(residuals, n_factors):
    random_state = np.random.RandomState(seed=42)
    baseline = random_state.random_sample()
    skater_scores = random_state.random_sample((residuals.shape[0], n_factors))
    event_scores = random_state.random_sample((n_factors, residuals.shape[1]))
    
    alpha = 0.0005  
    baseline_gradient = np.nansum(residuals)
    baseline = baseline - alpha * baseline_gradient    
      
    for k in range(n_factors):
        skater_scores_k = skater_scores[:, [k]]
        event_scores_k = event_scores[[k], :]
        
        event_gradients_k = np.nansum(residuals * skater_scores_k, axis=0, keepdims=True)
        skater_gradients_k = np.nansum(residuals * event_scores_k, axis=1, keepdims=True)    
                
        event_scores[[k], :] = event_scores_k - alpha * event_gradients_k
        skater_scores[:, [k]] = skater_scores_k - alpha * skater_gradients_k
    
    return baseline, event_scores, skater_scores

@profile
def broadcast_gradient_descent(residuals, n_factors):    
    random_state = np.random.RandomState(seed=42)
    baseline = random_state.random_sample()
    skater_scores = random_state.random_sample((residuals.shape[0], n_factors))
    event_scores = random_state.random_sample((n_factors, residuals.shape[1]))
    
    alpha = 0.0005
    baseline_gradient = np.nansum(residuals)
    baseline = baseline - alpha * baseline_gradient
    
    event_gradients = np.nansum(residuals[:, np.newaxis, :] * skater_scores[:, :, np.newaxis], axis=0)
    skater_gradients = np.nansum(residuals[:, np.newaxis, :] * event_scores[np.newaxis, :, :], axis=-1)
    
    event_scores = event_scores - alpha * event_gradients
    skater_scores = skater_scores - alpha * skater_gradients
    
    return baseline, event_scores, skater_scores


if __name__ == '__main__':
    residuals = np.load('viz/residuals.npy')
    naive_gradient_descent(residuals, 100000)
    broadcast_gradient_descent(residuals, 100000)