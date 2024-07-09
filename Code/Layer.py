from collections.abc import Callable
from Node import Node
import numpy as np

class Layer:
    """
    Abstract class for layers
    """
    def __init__(
            self,
            length: int,
            length_prev: int,
            act_func: Callable[[np.ndarray], np.ndarray],
            seed: int = 100
    ):
        self.nodes = []
        self.length = length
        self.length_prev = length_prev
        self.act_func = act_func
        self.seed = seed
        
        self.W_layer_size = (self.length_prev, self.length)
        self.b_layer_size = (1, self.length)
        self.W_time_size = (self.length, self.length)
        self.b_time_size = (1, self.length)

        self.n_nodes = 0
    
    def reset_weights(self):
        raise NotImplementedError
    
    def update_weights(self):
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