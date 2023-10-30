import numpy as np
from utils import *

'''This file contains the implementation code for the different models implemented.'''

class Generalised_Model:
    
    def __init__(self, loss_fnc_tag):
        
        """
        Args:
            loss_fnc_tag: tag for loss fnc to be used
        """
        self.loss_fnc_tag = loss_fnc_tag
        self.trained_weights = None
               
    def calc_acc(self, y_true, y_pred):
        
        """Calculates the accuracy of the trained model based on a threshold
        for the output probability.
        
        Args:
            y_true: numpy array of true y values with shape = (N,)
            y_pred: numpy array of predicted y values with shape = (N,)
            
        Returns:
            acc: accuracy
        """
        
        correct_pred_cnt = np.sum(y_true == y_pred)
        ttal_cnt = y_true.shape[0]
        return correct_pred_cnt/ ttal_cnt
    
    def calc_metrics(self, y_true, y_pred, f1_weight = 1.):
        
        """Calculates the recall, precision and F1 score
        
        Args:
            y_true: numpy array of true y values with shape = (N,)
            y_pred: numpy array of predicted y values with shape = (N,)
            
        Returns:
            recall, precision, weighted_f1
        """

        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        
        recall = tp/ (tp+fn) if tp+fn != 0 else 0.
        precision = tp/ (tp + fp) if tp+fp != 0 else 0.
        weighted_f1 = (1 + f1_weight**2)*((precision*recall)/(((f1_weight**2)*precision) + recall)) if ((f1_weight**2*precision) + recall) else 0
        
        return recall, precision, weighted_f1

    
class MeanSquared_Model(Generalised_Model):
    
    def __init__(self, loss_fnc_tag, training_fnc, calc_loss_fnc):
        self.training_fnc = training_fnc
        self.calc_loss_fnc = calc_loss_fnc
        super().__init__(loss_fnc_tag)
        
    def train(self, y, x, initial_w, max_iters, gamma):
        
        """Trains model and returns optimal weight and loss. 

        Args:
        y: numpy array of shape=(N, )
        x: numpy array of shape=(N,D)
        initial_w: numpy array of shape=(D,). The vector of model's initial parameters.
        max_iters: int. maximum number of iterations to train over
        gamma: float. step size in GD.

        Returns:
        w: numpy array of shape=(D,). The trained model parameters.
        loss: scalar. The calculated MSE loss from learnt parameters.
        
        """
        
        w, loss = self.training_fnc(y, x, initial_w, max_iters, gamma)    
        self.trained_weights = w
        
        return w, loss
            
    def calc_loss(self, y, x, w):
        
        if self.trained_weights is None:
            return self.calc_loss_fnc(y, x, w)
        
        return self.calc_loss_fnc(y, x, self.trained_weights)
    
    def predict(self, x, threshold = 0.5):
        
        """Get predictions for observations of y based on a threshold
        for the output probability.
        
        Args:
            y: numpy array of shape = (N,)
            x: numpy array of shape = (N, D)
            w: numpy vector of optimal weights = (D,)
            threshold: probability threshold to determine if predicted class is 0/1
            
        Returns:
            y_pred: numpy vector of predicted outputs of shape = (N,)
        """
        # Get logits
        y_pred = np.matmul(x, self.trained_weights)
    
        # Get probabilities for each y
        y_pred = np.where(y_pred > threshold, 1, 0)
        return y_pred
    
class LogReg_Model(Generalised_Model):
    
    def __init__(self, loss_fnc_tag, training_fnc, calc_loss_fnc):
        self.training_fnc = training_fnc
        self.calc_loss_fnc = calc_loss_fnc
        super().__init__(loss_fnc_tag)
        
    def train(self, y, x, initial_w, max_iters, gamma):
        
        """This function is to train the model to get the final trained weights and loss. Trains the model using regularised logistic regression using SGD where the process of applying gradient descent is encapsulated within self.training_fnc.

        Args:
            y: numpy array of shape=(N, )
            x: numpy array of shape=(N,D)
            initial_w: numpy array of shape=(D,). The vector of model's initial parameters.
            max_iters: int. maximum number of iterations to train over
            gamma: float. step size in GD.

        Returns:
            w: numpy array of shape=(D,). The trained model parameters.
            loss: scalar. The calculated MSE loss from learnt parameters.
        """
        
        w, loss = self.training_fnc(y, x, initial_w, max_iters, gamma)
        self.trained_weights = w
    
        return w, loss
    
    def train_withlc(self, train_y, train_x, test_y, test_x, initial_w, num_epoch, gamma):

        """This function is to train the model while receiving information about the evaluation metrics at every epoch. Trains the model using regularised logistic regression using SGD where the process of applying gradient descent is encapsulated within self.training_fnc.

        Args:
            train_y: numpy array of shape=(N_train, )
            train_x: numpy array of shape=(N_train,D)
            test_y: numpy array of shape=(N_test, )
            test_x: numpy array of shape=(N_test, D)
            initial_w: numpy array of shape=(D,). The vector of model's initial parameters
            num_epoch: int. maximum number of epochs to train over
            gamma: float. step size in GD.

        Returns:
            arrays of evaluation metrics for both train and test split
        """
            
        # Initialise weights
        self.w = initial_w

        # Arrays to store metrics
        train_loss = []
        train_acc = []
        train_f1 = []

        test_loss = []
        test_acc = []
        test_f1 = []
        
        for epoch in range(num_epoch):

            # Calculate gradients
            gradients = calculate_logreg_gradient(train_y, train_x, self.w)

            # Update weights
            self.w = self.w - gamma*gradients

            # Calculate loss and metrics on train data set
            tr_loss = self.calc_loss(train_y, train_x, self.w)
            
            train_y_pred = self.predict(train_x)
            tr_acc = self.calc_acc(train_y, train_y_pred)
            _, _, tr_f1 = self.calc_metrics(train_y, train_y_pred)
            
            train_loss.append(tr_loss)
            train_acc.append(tr_acc)
            train_f1.append(tr_f1)
            
            # Calculate loss and metrics on test data set
            te_loss = self.calc_loss(test_y, test_x, self.w)
            
            test_y_pred = self.predict(test_x)
            te_acc = self.calc_acc(test_y, test_y_pred)
            _, _, te_f1 = self.calc_metrics(test_y, test_y_pred)
            
            test_loss.append(te_loss)
            test_acc.append(te_acc)
            test_f1.append(te_f1)
            
        return train_loss, train_acc, train_f1, test_loss, test_acc, test_f1
            
    def calc_loss(self, y, x, w):

        if self.trained_weights is None:
            return self.calc_loss_fnc(y, x, w)

        return self.calc_loss_fnc(y, x, self.trained_weights)
    
    def predict(self, x, threshold = 0.5):

        """Get predictions for observations of y based on a threshold
        for the output probability.

        Args:
            y: numpy array of shape = (N,)
            x: numpy array of shape = (N, D)
            w: numpy vector of optimal weights = (D,)
            threshold: probability threshold to determine if predicted class is 0/1

        Returns:
            y_pred: numpy vector of predicted outputs of shape = (N,)
        """

        # Get logits
        
        # Used in train_withlc to get prediction at every epoch
        if self.trained_weights is None:
            y_pred = np.matmul(x, self.w)
        # Used in train to get prediction only at the ends
        else:
            y_pred = np.matmul(x, self.trained_weights)
        
        # Apply sigmoid 
        y_pred = sigmoid(y_pred)
        # Get probabilities for each y
        y_pred = np.where(y_pred > threshold, 1, 0)
        
        return y_pred
    
class LogReg_WithReg_Model(Generalised_Model):
    
    def __init__(self, loss_fnc_tag, training_fnc, calc_loss_fnc):
        self.training_fnc = training_fnc
        self.calc_loss_fnc = calc_loss_fnc
        super().__init__(loss_fnc_tag)
        
    def train(self, y, x, lambda_, initial_w, max_iters, gamma):
        
        """This function is to train the model to get the final trained weights and loss. Trains the model using regularised logistic regression using SGD where the process of applying gradient descent is encapsulated within self.training_fnc.

        Args:
            y: numpy array of shape=(N, )
            x: numpy array of shape=(N,D)
            initial_w: numpy array of shape=(D,). The vector of model's initial parameters.
            max_iters: int. maximum number of iterations to train over
            gamma: float. step size in GD.

        Returns:
            w: numpy array of shape=(D,). The trained model parameters.
            loss: scalar. The calculated MSE loss from learnt parameters.
        """
        
        w, loss = self.training_fnc(y, x, lambda_, initial_w, max_iters, gamma)
        self.trained_weights = w
    
        return w, loss
    
    def train_withlc(self, train_y, train_x, test_y, test_x, initial_w, num_epoch, lambda_, gamma):

        """This function is to train the model while receiving information about the evaluation metrics at every epoch. Trains the model using regularised logistic regression using SGD where the process of applying gradient descent is encapsulated within self.training_fnc.

        Args:
            train_y: numpy array of shape=(N_train, )
            train_x: numpy array of shape=(N_train,D)
            test_y: numpy array of shape=(N_test, )
            test_x: numpy array of shape=(N_test, D)
            initial_w: numpy array of shape=(D,). The vector of model's initial parameters
            num_epoch: int. maximum number of epochs to train over
            lambda_: float, value of regularisation term
            gamma: float. step size in GD.

        Returns:
            arrays of evaluation metrics for both train and test split
        """
            
        # Initialise weights
        self.w = initial_w

        # Arrays to store metrics
        train_loss = []
        train_acc = []
        train_f1 = []

        test_loss = []
        test_acc = []
        test_f1 = []

        for epoch in range(num_epoch):

            # Calculate gradients
            gradients = calculate_logreg_gradient(train_y, train_x, self.w) + (2*lambda_*self.w)

            # Update weights
            self.w = self.w - gamma*gradients

            # Calculate loss and metrics on train data set
            tr_loss = self.calc_loss(train_y, train_x, self.w)
            
            train_y_pred = self.predict(train_x)
            tr_acc = self.calc_acc(train_y, train_y_pred)
            _, _, tr_f1 = self.calc_metrics(train_y, train_y_pred)
            
            train_loss.append(tr_loss)
            train_acc.append(tr_acc)
            train_f1.append(tr_f1)
            
            # Calculate loss and metrics on test data set
            te_loss = self.calc_loss(test_y, test_x, self.w)
            
            test_y_pred = self.predict(test_x)
            te_acc = self.calc_acc(test_y, test_y_pred)
            _, _, te_f1 = self.calc_metrics(test_y, test_y_pred)
            
            test_loss.append(te_loss)
            test_acc.append(te_acc)
            test_f1.append(te_f1)
            
        return train_loss, train_acc, train_f1, test_loss, test_acc, test_f1
    
    def calc_loss(self, y, x, w):

        if self.trained_weights is None:
            return self.calc_loss_fnc(y, x, w)

        return self.calc_loss_fnc(y, x, self.trained_weights)
    
    def predict(self, x, threshold = 0.5):

        """Get predictions for observations of y based on a threshold
        for the output probability.

        Args:
            y: numpy array of shape = (N,)
            x: numpy array of shape = (N, D)
            w: numpy vector of optimal weights = (D,)
            threshold: probability threshold to determine if predicted class is 0/1

        Returns:
            y_pred: numpy vector of predicted outputs of shape = (N,)
        """

        # Get logits
        # Used in train_withlc to get prediction at every epoch
        if self.trained_weights is None:
            y_pred = np.matmul(x, self.w)
        # Used in train to get prediction only at the ends
        else:
            y_pred = np.matmul(x, self.trained_weights)
        
        # Apply sigmoid 
        y_pred = sigmoid(y_pred)
        # Get probabilities for each y
        y_pred = np.where(y_pred > threshold, 1, 0)
        
        return y_pred