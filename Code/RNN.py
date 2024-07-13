from collections.abc import Callable
from jax import vmap
from funcs import derivate
from sklearn.utils import resample
from schedulers import Scheduler
from Layer import Layer
from InputLayer import InputLayer
from OutputLayer import OutputLayer
from RNNLayer import RNNLayer
from Node import Node
import numpy as np

class RNN:
    def __init__(
            self,
            cost_func: Callable[
                [np.ndarray], # Takes in the target output (array)
                Callable[[np.ndarray], np.ndarray] # Returns a function (the cost function)
            ],
            seed: int = 100
    ):
        """
        cost_func = function which takes in the target array and returns a new function. The new function
            takes in the output from the network, and returns the cost of that output compared to the target.
        """
        self.layers = [] # List of layers
        self.n_layers = 0

        self.cost_func = cost_func
        self.seed = seed

        self.n_features_output = None
    
    def reset_weights(self):
        for layer in self.layers:
            layer.reset_weights()
    
    def reset_schedulers(self):
        for layer in self.layers:
            layer.reset_schedulers()

    def feed_forward(
            self,
            X: np.ndarray
    ):
        """
        X = input to RNN.
        X_shape = (batch_size, sequence_length, n_features)
        NOTE: Unlike the other layers, this layer takes a numpy array as input instead of a Layer.
        """
        X_shape = X.shape

        ## Initialize output
        n_batches = X_shape[0]
        sequence_length = X_shape[1]
        n_features_output = self.n_features_output
        output_shape = (n_batches, sequence_length, n_features_output)
        output = np.zeros(output_shape)

        self.layers[0].feed_forward(X)
        for i in range(1, self.n_layers):
            layer = self.layers[i]
            prev_layer = self.layers[i-1]
            layer.feed_forward(prev_layer)
        
        output_layer = layer
        for i in range(output_layer.n_nodes):
            node = output_layer.nodes[i]
            output[:,i,:] = node.h_output
        
        return output
    
    def backpropagate(
            self,
            output: np.ndarray,
            target: np.ndarray,
            lmbd: float = 0.01
    ):
        grad_cost = derivate(self.cost_func(target))
        dC = grad_cost(output)
        self.layers[-1].backpropagate(dC, lmbd)

        for i in range(self.n_layers-2, 0, -1):
            layer = self.layers[i]
            next_layer = self.layers[i+1]
            layer.backpropagate(next_layer, lmbd)
        
    def train(
            self,
            X_train: np.ndarray,
            t_train: np.ndarray,
            X_val: np.ndarray = None,
            t_val: np.ndarray = None,
            epochs: int = 100,
            batches: int = 1,
            lmbd: float = 0.01,
            store_output: np.ndarray = False
    ):
        """
        X = input (n_batches, sequence_length, n_features)
        t = target (n_batches, sequence_length, n_features)
        _train = training data
        _val = validation data
        epochs = number of epochs to train for
        batches = number of batches to split data into
        store_output = Whether to store output from each epoch. If true, will be saved in scores
            y_history shape = (n_epochs, n_batches, sequence_length, n_features)
        """
        self.reset_weights() # Reset weights for new training
        batch_size = X_train.shape[0] // batches

        # Initialize arrays for scores
        train_cost = self.cost_func(t_train)
        train_error = np.zeros(epochs)
        train_accuracy = np.zeros(epochs)

        if X_val is not None:
            val_cost = self.cost_func(t_val)
            val_error = np.zeros(epochs)
            val_accuracy = np.zeros(epochs)
        
        if store_output:
            y_train_shape = (epochs,) + X_train.shape
            y_train_history = np.zeros(shape=y_train_shape)
            if X_val is not None:
                y_val_shape = (epochs,) + X_val.shape
                y_val_history = np.zeros(shape=y_val_shape)
        
        # Resample X and t
        X_train, t_train = resample(X_train, t_train, replace=False)

        for e in range(epochs):
            print("EPOCH: " + str(e+1) + "/" + str(epochs), end="\r")
            for b in range(batches):
                ## Extract a smaller batch from the training data
                if b == batches - 1:
                    # If this is the last batch, include all remaining elements
                    X_batch = X_train[b*batch_size :]
                    t_batch = t_train[b*batch_size :]
                else:
                    X_batch = X_train[b*batch_size : (b+1)*batch_size]
                    t_batch = t_train[b*batch_size : (b+1)*batch_size]
                
                ## Train the network on this batch with gradient descent
                y_batch = self.feed_forward(X_batch)
                self.backpropagate(y_batch, t_batch, lmbd)
            
            self.reset_schedulers()

            ## Compute scores for this epoch
            y_train = self.feed_forward(X_train)
            train_error[e] = train_cost(y_train)

            if X_val is not None:
                y_val = self.feed_forward(X_val)
                val_error[e] = val_cost(y_val)
            
            if store_output:
                y_train_history[e,:,:,:] = y_train
                if X_val is not None:
                    y_val_history[e,:,:,:] = y_val
        
        ## Create a dictionary for the scores, and return it
        scores = {"train_error": train_error}
        if X_val is not None:
            scores["y_val"] = y_val
            scores["val_error"] = val_error
        
        if store_output:
            scores["y_train_history"] = y_train_history
            if X_val is not None:
                scores["y_val_history"] = y_val_history
        
        return scores

    def add_InputLayer(
            self,
            n_features: int
    ):
        """
        n_features = number of features expected from the input.
        """
        layer = InputLayer(n_features)
        self._add_layer(layer)
    
    def add_RNNLayer(
            self,
            n_features: int,
            act_func: Callable[[np.ndarray], np.ndarray],
            scheduler: Scheduler
    ):
        """
        n_features = number of features in this layer
        n_features_prev = number of features in the previous layer
        act_func = activation function for this layer
        seed = numpy random seed
        """
        prev_layer = self.layers[-1]
        n_features_prev = prev_layer.n_features
        layer = RNNLayer(n_features, n_features_prev, act_func, scheduler, self.seed)
        self._add_layer(layer)

    def add_OutputLayer(
            self,
            n_features: int,
            act_func: Callable[[np.ndarray], np.ndarray],
            scheduler: Scheduler
    ):
        """
        n_features = number of features in this layer
        n_features_prev = number of features in the previous layer
        act_func = activation function for this layer
        seed = numpy random seed
        """
        prev_layer = self.layers[-1]
        n_features_prev = prev_layer.n_features
        layer = OutputLayer(n_features, n_features_prev, act_func, scheduler, self.seed)
        self._add_layer(layer)
        
        self.n_features_output = n_features
    
    def _add_layer(
            self,
            layer: Layer
    ):
        self.layers.append(layer)
        self.n_layers += 1