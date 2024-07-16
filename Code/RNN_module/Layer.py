from __future__ import annotations # Necessary to create typing hint of Layer within the class Layer
from collections.abc import Callable
from .Node import Node
import numpy as np

class Layer:
    """
    Abstract class for layers
    """
    def __init__(
            self,
            n_features: int,
            n_features_prev: int,
            act_func: Callable[[np.ndarray], np.ndarray],
            seed: int = 100
    ):
        self.nodes = []
        self.n_features = n_features
        self.n_features_prev = n_features_prev
        self.act_func = act_func
        self.seed = seed
        
        self.W_layer_size = (self.n_features_prev, self.n_features)
        self.b_layer_size = (1, self.n_features)
        self.W_time_size = (self.n_features, self.n_features)
        self.b_time_size = (1, self.n_features)

        self.n_nodes = 0
    
    def reset_weights(self):
        raise NotImplementedError
    
    def reset_schedulers(self):
        raise NotImplementedError
    
    def update_weights_all_nodes(self):
        raise NotImplementedError
    
    def add_node(self):
        raise NotImplementedError
    
    def remove_nodes(self):
        raise NotImplementedError

    def feed_forward(
            self,
            prev_layer: Layer
    ):
        raise NotImplementedError
    
    def backpropagate(
            self,
            next_layer: Layer,
            lmbd: float
    ):
        raise NotImplementedError