from Layer import Layer
from Node import Node
import numpy as np
 
class RNNLayer(Layer):
    def __init__(
            self,
            W_layer: np.ndarray,
            b_layer: np.ndarray,
            W_time: np.ndarray,
            b_time: np.ndarray
    ):
        """
        W = weights, b = bias
        layer = from previous layer to this one
        time = between time steps in the layer
        """
        super(W_layer, b_layer, W_time, b_time)
    
    def initialize_weights(self):
        pass

    def update_weights_all_nodes(
            self,
            new_W_layer: np.ndarray,
            new_b_layer: np.ndarray,
            new_W_time: np.ndarray,
            new_b_time: np.ndarray
    ):
        pass

    def feed_forward(
            self,
            X: np.ndarray
    ):
        pass
    
    def backpropagate(
            self,
            dC: np.ndarray
    ):
        pass