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
            seed: int = 100
    ):
        self.n_features = n_features
        self.seed = seed
        self.nodes = []
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