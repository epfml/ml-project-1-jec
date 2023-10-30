"""
This python file provides all the implentations for the different loss functions with the specified signature. Utility functions are implemented in the utils.py file.

Notes about the loss functions to be written:

1. In the above method signatures, for iterative methods, initial w is the initial weight vector, gamma is the step-size, and max iters is the number of steps to run. lambda is always the regularization parameter.

2. For SGD, you must use the standard mini-batch-size 1.  (sample just one datapoint)

3. The mean squared error formula has a factor 0.5 to be consistent with the lecture notes.

4. All vectors should be implemented as 1D arrays with shape (X,) instead of (X, 1).

5. Return type 
    -  All functions should return: (w, loss), which is the last weight vector of the method, and the corresponding loss value (cost function).
    - Loss returned by the regularized methods (ridge regression and reg logistic regression) should not include the penalty term.
    - For functions performed without iterations (without max_iters argument), simply compute optimal weight and corresponding loss. Otherwise, just need to provide final weights and loss after going through the max. number of iterations.
"""

import numpy as np
from utils import *

def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    
    """Trains the model using MSE with GD.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,D)
        initial_w: numpy array of shape=(D,). The vector of model's initial parameters.
        max_iters: int. maximum number of iterations to train over
        gamma: float. step size in GD.

    Returns:
        w: numpy array of shape=(D,). The trained model parameters.
        loss: scalar. The calculated MSE loss from learnt parameters.
    """
        
    w = initial_w
    
    for n_iter in range(max_iters):
        
        gradients = compute_mse_gradient(y, tx, w)
        
        # Update weights
        w = w - gamma*gradients
         
    loss = compute_mse_loss(y, tx, w)
    
    # return trained weights and loss with trained weights
    return w, loss

def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    
    """Trains the model using MSE with SGD. This function is similar to mean_squared_error_gd() but SGD is used instead.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,D)
        initial_w: numpy array of shape=(D,). The vector of model's initial parameters.
        max_iters: int. maximum number of iterations to train over
        gamma: float. step size in GD.

    Returns:
        w: numpy array of shape=(D,). The trained model parameters.
        loss: scalar. The calculated MSE loss from learnt parameters.
    """
    
    # As per project description requirements
    batch_size = 1
    w = initial_w

    # Get batched data
    batched_data = list(batch_iter(y, tx, batch_size, num_batches = max_iters))
    cur_batched_idx = 0
    
    for n_iter in range(max_iters):
        
        # If have reached the end of the dataset, bring index back to the start
        if n_iter % len(batched_data) == 0:
            cur_batched_idx = 0
                    
        batched_y, batched_tx = batched_data[cur_batched_idx]
        
        gradients = compute_mse_stoch_gradient(batched_y, batched_tx, w)
        
        # Update weights
        w = w - gamma * gradients
        
        # Update cur_batched_idx
        cur_batched_idx += 1
        
    loss = compute_mse_loss(batched_y, batched_tx, w)
    
    return w, loss

def least_squares(y, tx):
    
    """Trains the model using least squares.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,D)

    Returns:
        w: numpy array of shape=(D,). The trained model parameters.
        loss: scalar. The calculated MSE loss from learnt parameters.
    """

    w = np.matmul(np.linalg.inv(np.matmul(tx.T, tx)), tx.T).dot(y)
    e = y - tx.dot(w)
    N = y.shape[0]
    loss = (1/(2*N))*(e.T.dot(e))

    return w, loss

def ridge_regression(y, tx, lambda_):
    
    """Trains the model using ridge regression (Least Squares with penalty).

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,D)
        initial_w: numpy array of shape=(D,). The vector of model's initial parameters.
        max_iters: int. maximum number of iterations to train over
        gamma: float. step size in GD.

    Returns:
        w: numpy array of shape=(D,). The trained model parameters.
        loss: scalar. The calculated MSE loss from learnt parameters.
    """

    
    # Use 2N here to be consistent with lecture notes
    # Calculate optimal weights
    num_samples = y.shape[0]
    num_params = tx.shape[1]
    inverse_product = np.linalg.inv(np.matmul(tx.T, tx) + (2*num_samples*lambda_)*np.identity(num_params))
    optimal_w = np.matmul(inverse_product, tx.T).dot(y)
    
    # Calculate loss
    optimal_e = y - tx.dot(optimal_w)
    optimal_loss = (1/(2*num_samples))*(optimal_e.T.dot(optimal_e)) # loss does not include penalty term

    return optimal_w, optimal_loss


# Note: GD will be used as the test script was based on GD
def logistic_regression(y, tx, initial_w, max_iters, gamma):
    
    """Trains the model using regularised logistic regression using SGD.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,D)
        initial_w: numpy array of shape=(D,). The vector of model's initial parameters.
        max_iters: int. maximum number of iterations to train over
        gamma: float. step size in GD.

    Returns:
        w: numpy array of shape=(D,). The trained model parameters.
        loss: scalar. The calculated MSE loss from learnt parameters.
    """

    w = initial_w
            
    # Iterate through till max_iters
    for n_iter in range(max_iters):
        
        # Calculate the gradients
        gradient = calculate_logreg_gradient(y, tx, w)
        
        # Update the weights
        w = w - gamma*gradient
        
    # Get final loss based on weights
    loss = calculate_logreg_loss(y, tx, w)
    
    # Return loss and weights
    return w, loss


def reg_logistic_regression(y, tx, lambda_ ,initial_w, max_iters, gamma):

    """Trains the model using regularised logistic regression using GD. The loss returned by regularised methods do not include the penalty term.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,D)
        initial_w: numpy array of shape=(D,). The vector of model's initial parameters.
        max_iters: int. maximum number of iterations to train over
        gamma: float. step size in GD.

    Returns:
        w: numpy array of shape=(D,). The trained model parameters.
        loss: scalar. The calculated MSE loss from learnt parameters.
    """
    # NOTE: loss returned by the regularized methods should not include the penalty term
    
    w = initial_w
        
    # Iterate through till max_iters
    for n_iter in range(max_iters):
                
        # Calculate the gradients
        gradient = calculate_logreg_gradient(y, tx, w) + (2*lambda_*w) 
        
        # Update the weights
        w = w - gamma*gradient
        
    # Get final loss based on weights
    # Note: Check if final loss should be calculated on entire dataset or just one point only
    loss = calculate_logreg_loss(y, tx, w)
    
    # Return loss and weights
    return w, loss


