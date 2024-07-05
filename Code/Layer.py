from Node import Node
import numpy as np

class Layer:
    """
    Abstract class for layers
    """
    def __init__(
            self,
            W_layer: np.ndarray,
            b_layer: np.ndarray,
            W_time: np.ndarray,
            b_time: np.ndarray
    ):
        self.nodes = []
        self.W_layer = W_layer
        self.b_layer = b_layer
        self.W_time = W_time
        self.b_time = b_time
    
    def initialize_weights(self):
        raise NotImplementedError
    
    def update_weights_all_nodes(
            self,
            new_W_layer: np.ndarray,
            new_b_layer: np.ndarray,
            new_W_time: np.ndarray,
            new_b_time: np.ndarray
    ):
        raise NotImplementedError

    def feed_forward(
            self,
            X: np.ndarray
    ):
        raise NotImplementedError
    
    def backpropagate(
            self,
            dC: np.ndarray
    ):
        raise NotImplementedError