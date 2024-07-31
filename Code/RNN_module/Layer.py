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
        """
        Remove all the nodes created for this layer.

        NOTE
        ----
        The weights and biases of the nodes are still stored in the layer, so we can easily
            create new nodes. Removing the nodes is used to allow the sequence length to vary with
            each call of feed_forward().
        """
        self.nodes = []
        self.n_nodes = 0

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