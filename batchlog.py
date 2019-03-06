import numpy as np


class BatchLogistic:
    def __init__(self, theta, alpha, lambda_reg=0):
        # Initialize model object with given theta, alpha (learning rate),
        # and lambda_reg (regularization hyperparameter)
        self.theta = np.array(theta)
        self.alpha = alpha
        self.lambda_reg = lambda_reg
        
    def sigmoid(self, X, theta):
        # Sigmoid function to calculate probability of weight gain from feature matrix X and theta vector
        return 1 / (1 + np.exp(-X @ theta))
    
    def fit(self, X, y, n_iter):
        self.avg_log_likelihoods = []
        self.thetas = []
        
        for i in range(n_iter):
            # Record theta for every iteration
            self.thetas.append(self.theta)
            
            prob = self.sigmoid(X, self.theta) # Step 1      
            # Record average log-likelihood for every iteration
            if np.all(prob != 0) or np.all(prob != 1):
                self.avg_log_likelihood = (y @ np.log(prob) + (1 - y) @ np.log(1 - prob)) / len(y)
                self.avg_log_likelihoods.append(self.avg_log_likelihood)
            
            # Calculate regularization term for ridge regression
            reg_term = self.lambda_reg * self.theta
            # First feature (intercept) is not regularized
            reg_term[0] = 0
            
            self.gradient = (y - prob) @ X - reg_term # Step 2, note the extra reg_term subtracted at the end
            self.theta = self.theta + self.alpha * self.gradient # Step 3
        # Record difference in average log-likelihood for the last iteration
        self.last_avg_log_likelihood_diff = self.avg_log_likelihoods[-1] - self.avg_log_likelihoods[-2]
        self.thetas = np.array(self.thetas)
            
    def predict(self, X, threshold=0.5):
        # Return predicted labels when using trained model on X feature matrix
        return (self.sigmoid(X, self.theta) > threshold).astype(int)
    
    def predict_proba(self, X):
        # Return predicted probability of weight gain using trained model on X feature matrix
        return self.sigmoid(X, self.theta)