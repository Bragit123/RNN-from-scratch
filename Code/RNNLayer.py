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