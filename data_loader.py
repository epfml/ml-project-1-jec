'''This python file preprocesses the data sets for the input features and the response variables and loads them into numpy array to be used for training by the models.'''

import os
import numpy as np

from provided_helpers import *

#def one_hot(a, num_classes):
#  return np.squeeze(np.eye(num_classes)[a.reshape(-1)])

def one_hot(a, num_classes):
    return np.squeeze(np.eye(num_classes)[a.reshape(-1)]).astype(int)
    
def load_x_data(in_path):
    
    # Load in their numpy file using their helper function
    x_np = np.load(in_path)
    
    # Determine the indexes of the selected columns
    # selected columns for Model 1, 2 and 3
    #selected_cols = ['_RFHYPE5', '_RFHLTH', '_AGE65YR', 'CVDSTRK3', 'CHCCOPD1', 'QLACTLM2', 'DIABETE3', '_RFCHOL', '_FLSHOT6', '_LMTACT1', '_PNEUMO2', '_HCVU651']
    #indices = [232, 230, 247, 39, 44, 65, 48, 234, 318, 313, 319, 231]
    
    # selected columns for Model 4 and 5
    selected_cols = ['_RFHYPE5', '_RFHLTH', '_AGE65YR', 'CVDSTRK3', 'CHCCOPD1', 'QLACTLM2', 'DIABETE3', '_RFCHOL', 'HLTHPLN1']
    indices = [232, 230, 247, 39, 44, 65, 48, 234, 30]
    x_np = x_np[:, indices]
    
    # Fill in NaN values with 7
    x_np[np.isnan(x_np)] = 7
    x_np[x_np == 9] = 7

    # Data preprocessing for _AGE65YR and DIABETE3 columns
    age65_col = x_np[:,2]
    age65_col[age65_col == 3] = 7
    x_np[:,2] = age65_col

    diabete3_col = x_np[:,6]
    diabete3_col[diabete3_col == 2] = 3
    diabete3_col[diabete3_col == 4] = 3
    x_np[:, 6] = diabete3_col

    # Create an empty list
    processed_x_np = []

    # For each of the selected column, generate the embedding columns for that row
    for i in range(x_np.shape[1]):

        # Map values to the range [0, num_classes-1]
        unique_vals = np.unique(x_np[:, i])
        value_to_index = {value: idx for idx, value in enumerate(unique_vals)}
        x_mapped = np.array([value_to_index[value] for value in x_np[:, i]])
        
        num_classes = len(unique_vals)  # Use the actual number of unique values
        encoded_col_np = one_hot(x_mapped, num_classes)

        # Suppress last level
        encoded_col_np = np.delete(encoded_col_np, -1, axis=1)

        for j in range(encoded_col_np.shape[1]):
            processed_x_np.append(encoded_col_np[:,j])

    # Make encoded columns into a numpy matrix
    processed_x_np = np.column_stack(processed_x_np)
    
    # Add intercept column to the start of the numpy array
    intercept_col = np.ones((processed_x_np.shape[0], 1), dtype=float)
    processed_x_np = np.hstack((intercept_col, processed_x_np))

    # Return the numpy array
    return processed_x_np
    
def load_y_data(in_path, flag):
    
    """This function reformats the y labels from {-1,1} to {0,1} and vice versa
    
    Args:
        in_path: file path to input y np file
        out_path: file path to output y np file
        flag: True if converting from {-1,1} to {0,1} and False for the other way

    Returns:
        y_np: numpy array of shape (N,) where N is number of observations with formatted y data
    """
    
    if not os.path.isfile(in_path):
        raise FileNotFoundError
        
    y_np = np.load(in_path)
    
    if flag:
        y_np[y_np == -1] = 0
    else:
        y_np[y_np == 0] = -1
    
    return y_np