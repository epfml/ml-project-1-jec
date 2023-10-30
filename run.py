'''This python file generates the csv file for submission. The csv file is saved as 'submission.csv'.'''

# Script to generate submission
import math
from implementations import *
from utils import *
from data_loader import *
from provided_helpers import *
from models import *

# Function to downsample dataset
def downsample_ds(x_train, y_train, pos_proportion = 0.5):
    
    """Downsamples the data to achieve a desired proportion of observations of the different classes.

    Args:
        x_train: numpy array of x_train data with shape (N, D)
        x_test: numpy array of x_test data with shape (N, D)
        y_train: numpy array of y_train data with shape (N,)
        pos_proportion: desired proportion of observations from positive class
        
    Returns:
        x_train, x_test and y_train with the desired proportions
    """    
    
    # Get number of positive and negative observations
    num_positive_obs = np.sum(y_train == 1)
    num_negative_obs = math.floor((num_positive_obs/pos_proportion) - num_positive_obs)
 
    # Get the positive observations
    positive_obs_idxs = np.where(y_train == 1)[0]
    x_sampled_positive_obs = x_train[positive_obs_idxs]
    y_sampled_positive_obs = y_train[positive_obs_idxs]
   
    # Get the negative observations
    # Randomly permutate the indexes of negative observations
    negative_obs_idxs = np.where(y_train == 0)[0]
    np.random.shuffle(negative_obs_idxs)
    
    # Get the desired numberof negative observations
    sampled_negative_obs_idxs = negative_obs_idxs[:num_negative_obs]
    x_sampled_negative_obs = x_train[sampled_negative_obs_idxs] 
    y_sampled_negative_obs = y_train[sampled_negative_obs_idxs]
    
    # Combine positive and negative samples to get downsampled dataset
    x_train_downsampled = np.append(x_sampled_positive_obs, x_sampled_negative_obs, axis = 0)
    y_train_downsampled = np.append(y_sampled_positive_obs, y_sampled_negative_obs)
    
    return x_train_downsampled, y_train_downsampled

# To generate submission csv file
# Instantiate model
model = LogReg_WithReg_Model('logreg_withreg_gd', reg_logistic_regression, calculate_logreg_loss)

x_train_np = load_x_data('datasets/updated_ds/x_train.npy')
x_test_np = load_x_data('datasets/updated_ds/x_test.npy')

y_train = load_y_data('datasets/updated_ds/y_train.npy', True)

# Downsample dataset
x_train_downsampled, y_train_downsampled = downsample_ds(x_train_np, y_train)

# Train the model
initial_w = np.zeros((x_train_downsampled.shape[1],), dtype = np.float64)
max_iters = 100
gamma = 0.1
lambda_ = 8.25
weights, _ = model.train(y_train_downsampled, x_train_downsampled, lambda_, initial_w, max_iters, gamma)

# Get the predictions for x_test
y_test_pred = model.predict(x_test_np)

# Format to {-1, 1} from {0, 1}.
y_test_pred[y_test_pred == 0] = -1

# Get test observations ids
ids = np.load('datasets/updated_ds/test_ids.npy')

# Generate csv_submission
create_csv_submission(ids, y_test_pred, 'submission.csv')
