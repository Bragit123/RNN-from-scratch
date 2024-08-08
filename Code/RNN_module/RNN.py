from collections.abc import Callable
from jax import vmap
from sklearn.utils import resample
from copy import copy
from .funcs import derivate
from .schedulers import Scheduler
from .Layer import Layer
from .InputLayer import InputLayer
from .OutputLayer import OutputLayer
from .DenseLayer import DenseLayer
from .RNNLayer import RNNLayer
from .Node import Node
import numpy as np

class RNN:
    def __init__(
            self,
            cost_func: Callable[
                [np.ndarray], # Takes in the target output (array)
                Callable[[np.ndarray], np.ndarray] # Returns a function (the cost function)
            ],
            scheduler: Scheduler,
            seed: int = 100
    ):
        """
        cost_func = function which takes in the target array and returns a new function. The new function
            takes in the output from the network, and returns the cost of that output compared to the target.
        """
        self.layers = [] # List of layers
        self.n_layers = 0

        self.cost_func = cost_func
        self.scheduler = scheduler
        self.seed = seed
        self.single_output = None # Boolean. Will update this when adding OutputLayer or DenseLayer output

        self.n_features_output = None
        self.output = None
        self.predicted = None
        self.output_extra = None
    
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
        output_shape = X_shape if OutputLayer; (batch_size, n_features) if SingleOutputLayer
        """
        X_shape = X.shape

        ## Initialize output
        n_batches = X_shape[0]
        sequence_length = X_shape[1]
        n_features_output = self.n_features_output
        
        if self.single_output:
            output_shape = (n_batches, n_features_output)
            output = np.zeros(output_shape)
        else:
            output_shape = (n_batches, sequence_length, n_features_output)
            output = np.zeros(output_shape)
            
        ## Feed forward through all layers
        self.layers[0].feed_forward(X)
        for i in range(1, self.n_layers):
            layer = self.layers[i]
            prev_layer = self.layers[i-1]
            layer.feed_forward(prev_layer)
        
        ## Get output from last layer
        output_layer = layer
        if output_layer.is_dense:
            node = output_layer.nodes[0]
            output = node.h_output
        else:
            for i in range(output_layer.n_nodes):
                node = output_layer.nodes[i]
                output[:,i,:] = node.h_output
        
        ## Store and return output
        self.output = output
        return self.output
    
    def predict(
            self,
            X: np.ndarray
    ):
        """
        X = input to RNN.
        X_shape = (batch_size, sequence_length, n_features)
        output_shape = X_shape if OutputLayer; (batch_size, n_features) if DenseLayer as output
        """
        ## Initialize predicted array
        output = self.feed_forward(X)
        predicted = np.zeros_like(output)

        ## Find maximum values at the feature axis, and set predicted to 1 at those indices
        ind_batch = np.arange(predicted.shape[0])
        if self.single_output:
            ind_max = np.argmax(output, axis=1)

            # Set predicted to 1 at the maximum values
            predicted[ind_batch, ind_max] = 1
        else:
            ind_seq = np.arange(output.shape[1])
            ind_max = np.argmax(output, axis=2)
            ind_seq, ind_batch = np.meshgrid(ind_seq, ind_batch)
            
            # Set predicted to 1 at the maximum values
            predicted[ind_batch, ind_seq, ind_max] = 1
        
        self.predicted = predicted
        return self.predicted


    def extrapolate(
            self,
            length: int
    ):
        """
        Extrapolate data by continuing the sequence from the output of the last layer, with the output of the previous time step as input for the new time step
        length = number of time steps to extrapolate
        """
        ## Initialize extrapolated output
        output = self.output
        n_batches = output.shape[0]
        n_features = self.n_features_output
        output_extra_shape = (n_batches, length, n_features)
        output_extra = np.zeros(output_extra_shape)

        ## Extrapolate
        y_prev = output[:,0,:] # Output from last layer at last time step

        for i in range(length):
            h_layer = y_prev # h_layer at first layer = input = previous output
            for l in range(1, self.n_layers):
                layer = self.layers[l]
                node_prev = layer.nodes[-1] # Node at previous time step
                h_time = node_prev.h_output
                # print(h_layer, h_time)
                layer.add_node()
                node = layer.nodes[-1] # Current node
                if l == self.n_layers-1:
                    # If this is the output layer (last layer), set h_time to None
                    h_time = None
                node.feed_forward(h_layer, h_time)
                h_layer = node.h_output # Update h_layer
            y_prev = h_layer
            output_extra[:,i,:] = y_prev
        
        self.output_extra = output_extra
        return self.output_extra

    
    def backpropagate(
            self,
            output: np.ndarray,
            target: np.ndarray,
            lmbd: float = 0.01
    ):
        """
        output_shape = (n_batches, sequence_length, n_features) if OutputLayer;
            (n_batches, n_features) if SingleOutputLayer
        target_shape = same as output_shape
        """
        ## Find gradient of cost function
        grad_cost = derivate(self.cost_func(target))
        dC = grad_cost(output)

        ## Backpropagate through all layers
        self.layers[-1].backpropagate(dC, lmbd=lmbd)
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
        
        ## Initialize arrays for output history if this should be stored
        if store_output:
            n_batches_train = X_train.shape[0]
            seq_length_train = X_train.shape[1]
            n_features_output = self.n_features_output

            if self.single_output:
                y_train_shape = (epochs, n_batches_train, n_features_output)
            else:
                y_train_shape = (epochs, n_batches_train, seq_length_train, n_features_output)
            
            y_train_history = np.zeros(shape=y_train_shape)
            
            if X_val is not None:
                n_batches_val = X_val.shape[0]
                seq_length_val = X_val.shape[1]

                if self.single_output:
                    y_val_shape = (epochs, n_batches_val, n_features_output)
                else:
                    y_val_shape = (epochs, n_batches_val, seq_length_val, n_features_output)
                
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
            pred_train = self.predict(X_train)

            train_error[e] = train_cost(y_train)
            
            train_acc_arr = np.all(pred_train == t_train, axis=-1)
            train_accuracy[e] = np.mean(train_acc_arr)

            if X_val is not None:
                y_val = self.feed_forward(X_val)
                pred_val = self.predict(X_val)
                
                val_error[e] = val_cost(y_val)
            
                val_acc_arr = np.all(pred_val == t_val, axis=-1)
                val_accuracy[e] = np.mean(val_acc_arr)
            
            if store_output:
                y_train_history[e] = y_train
                if X_val is not None:
                    y_val_history[e] = y_val
        
        ## Create a dictionary for the scores, and return it
        scores = {"train_error": train_error, "train_accuracy": train_accuracy}
        if X_val is not None:
            scores["y_val"] = y_val
            scores["val_error"] = val_error
            scores["val_accuracy"] = val_accuracy
        
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
        layer = InputLayer(n_features, self.seed)
        self._add_layer(layer)
    
    def add_RNNLayer(
            self,
            n_features: int,
            act_func: Callable[[np.ndarray], np.ndarray]
    ):
        """
        n_features = number of features in this layer
        n_features_prev = number of features in the previous layer
        act_func = activation function for this layer
        seed = numpy random seed
        """
        scheduler = copy(self.scheduler)
        prev_layer = self.layers[-1]
        n_features_prev = prev_layer.n_features
        layer = RNNLayer(n_features, n_features_prev, act_func, scheduler, self.seed)
        self._add_layer(layer)

    def add_OutputLayer(
            self,
            n_features: int,
            act_func: Callable[[np.ndarray], np.ndarray]
    ):
        """
        n_features = number of features in this layer
        n_features_prev = number of features in the previous layer
        act_func = activation function for this layer
        seed = numpy random seed
        """
        scheduler = copy(self.scheduler)
        prev_layer = self.layers[-1]
        n_features_prev = prev_layer.n_features
        layer = OutputLayer(n_features, n_features_prev, act_func, scheduler, self.seed)
        self._add_layer(layer)
        
        self.single_output = False
        self.n_features_output = n_features
    
    def add_DenseLayer(
            self,
            n_features: int,
            act_func: Callable[[np.ndarray], np.ndarray],
            is_last_layer: bool = False
    ):
        scheduler = copy(self.scheduler)
        prev_layer = self.layers[-1]
        n_features_prev = prev_layer.n_features
        layer = DenseLayer(n_features, n_features_prev, act_func, scheduler, self.seed)
        self._add_layer(layer)

        if is_last_layer:
            self.single_output = True
            self.n_features_output = n_features

    def _add_layer(
            self,
            layer: Layer
    ):
        self.layers.append(layer)
        self.n_layers += 1