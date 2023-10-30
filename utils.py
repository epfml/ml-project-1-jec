"""

This python file provides all utility function to help with the implementation of batching of data for SGD and the calculation of losses

"""
import numpy as np

def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            # yield keyword will turn any expression that is given with it into a generator object and return it to the caller
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]
            

def compute_mse_loss(y, tx, w):
    """Calculate the loss using either MSE or MAE.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,D)
        w: numpy array of shape=(D,). The vector of model parameters.

    Returns:
        the value of the loss (a scalar), corresponding to the input parameters w.
    """
    
    N = y.shape[0]
    err = y - tx.dot(w)
    return (1/(2*N))*(err.T.dot(err))


def compute_mse_gradient(y, tx, w):
    """Computes the gradient at w.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,D)
        w: numpy array of shape=(D, ). The vector of model parameters.

    Returns:
        An numpy array of shape (D, ) (same shape as w), containing the gradient of the loss at w.
    """
    
    N = y.shape[0]
    e = y - tx.dot(w)
    return (-1/N)*(tx.T.dot(e))


def compute_mse_stoch_gradient(y, tx, w):
    
    """Compute a stochastic gradient at w from just few examples n and their corresponding y_n labels.

    Args:
        y: numpy array of shape=(B, )
        tx: numpy array of shape=(B,D)
        w: numpy array of shape=(D, ). The vector of model parameters.
        
        where B is the batch size

    Returns:
        A numpy array of shape (D, ) (same shape as w), containing the stochastic gradient of the loss at w.
        
    Additional notes:
        While the compute_mse_gradient will derive the same answer via matrix operations, the code written below is   adapted to follow the definition of mini-batch SGD where the mini-batch gradient is the average of gradients obtained for each point in the batch. 
    """
        
    # array to store results
    batch_size, num_params = y.shape[0], tx.shape[1]
    gradients = np.zeros((batch_size, num_params), dtype = np.float64)

    for idx in range(batch_size):
        
        # using previous compute_gradient function
        gradients[idx,:] = compute_mse_gradient(np.array([y[idx]]), np.array([tx[idx, :]]), w)
        
    # take average of gradients
    return np.average(gradients, axis = 0)


def sigmoid(t):
    
    """apply sigmoid function on t.

    Args:
        t: scalar or numpy array

    Returns:
        scalar or numpy array
    """
    
    t = (np.exp(t))/ (1 + np.exp(t))
    return t


def calculate_logreg_loss(y, tx, w):
    """compute the cost by negative log likelihood.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1)

    Returns:
        a non-negative loss
    """
    # assert y.shape[0] == tx.shape[0]
    # assert tx.shape[1] == w.shape[0]
    
    N = y.shape[0]
    txdotw = tx.dot(w)
    txdotw = np.squeeze(txdotw)
    sig_txdotw = sigmoid(txdotw)
    
    y = np.squeeze(y)
    return (-1/N)*((np.matmul(y,np.log(sig_txdotw))) + (np.matmul(np.ones(y.shape) - y,
                                                       (np.log(1-sig_txdotw)))))

    
def calculate_logreg_gradient(y, tx, w):
    
    """compute the gradient of loss.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1)

    Returns:
        a vector of shape (D, 1)
    """
    
    txdotw = tx.dot(w)
    sig_txdotw = sigmoid(txdotw)
    N = y.shape[0]
    
    return (1/N)*(np.matmul(tx.T,sig_txdotw - y))

                
def calculate_logreg_stoch_gradient(y, tx, w):
    
    """Compute the gradient of loss.

    Args:
        y:  shape=(B, 1)
        tx: shape=(B, D)
        w:  shape=(D, 1)
        
        where B is the batch size
        
    Returns:
        a vector of shape (D, 1)
        
    Additional notes:
        While the compute_mse_gradient will derive the same answer via matrix operations, the code written below is   adapted to follow the definition of mini-batch SGD where the mini-batch gradient is the average of gradients obtained for each point in the batch. 
    """
    
    # array to store results
    batch_size, num_params = y.shape[0], tx.shape[1]
    gradients = np.zeros((batch_size, num_params), dtype = np.float64)

    for idx in range(batch_size):
        
        # using previous compute_gradient function
        gradients[idx,:] = calculate_logreg_gradient(np.array([y[idx]]), np.array([tx[idx, :]]), w)
        
    # take average of gradients
    return np.average(gradients, axis = 0)