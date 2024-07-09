from collections.abc import Callable
from Layer import Layer
from Node import Node
import numpy as np

class InputLayer(Layer):
    def __init__(
            self,
            X: np.ndarray
    ):
        """
        X = input to RNN.
        X_shape = (batch_size, sequence_length, num_features)
        """
        self.nodes = []
        X_shape = X.shape
        sequence_length = X_shape[1]
        self.length = X_shape[2]
        self.n_nodes = 0

        # Add a node to the layer for each time step, and set output
        for i in range(sequence_length):
            self.add_node()
            self.nodes[i].set_output(X[:,i,:])
    
    def reset_weights(self):
        """
        This method should not be called in the input layer.
        """
        print("WARNING: reset_weights() was called in InputLayer. This should not be necessary.")
    
    def update_weights(self):
        """
        This method should not be called in the input layer.
        """
        print("WARNING: update_weights() was called in InputLayer. This should not be necessary.")
    
    def update_weights_all_nodes(
            self,
            new_W_layer: np.ndarray,
            new_b_layer: np.ndarray,
            new_W_time: np.ndarray,
            new_b_time: np.ndarray
    ):
        """
        This method should not be called in the input layer.
        """
        print("WARNING: update_weights_all_nodes() was called in InputLayer. This should not be necessary.")
    
    def add_node(self):
        """
        Add a node. Weights and biases are set to None, and activation to identity by default,
        as none of these are relevant for the input layer.
        """
        new_node = Node(self.length) # Activation and weights are not used for the input layer
        self.nodes.append(new_node)
        self.n_nodes += 1

    def remove_nodes(self):
        """
        Remove all the nodes created for this layer.
        """
        self.nodes = []
        self.n_nodes = 0

    def feed_forward(
            self,
            prev_layer: Layer
    ):
        """
        This method should not be called in the input layer.
        """
        print("WARNING: feed_forward() was called in InputLayer. This should not be necessary.")
    
    def backpropagate(
            self,
            dC: np.ndarray
    ):
        """
        This method should not be called in the input layer.
        """
        print("WARNING: backpropagate() was called in InputLayer. This should not be necessary.")