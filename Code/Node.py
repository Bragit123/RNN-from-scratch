import numpy as np

class Node:
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
        time = from previous time step to this one
        """
        self.W_layer = W_layer
        self.b_layer = b_layer
        self.W_time = W_time
        self.b_time = b_time
    
    def set_Wb(
            self,
            W_layer: np.ndarray,
            b_layer: np.ndarray,
            W_time: np.ndarray,
            b_time: np.ndarray
    ):
        pass

    def feed_forward(
            self,
            h_layer: np.ndarray,
            h_time: np.ndarray
    ):
        """
        h_layer/h_time = output from node at previous layer/time
        """
        pass

    def backpropagate(
            self,
            dC_layer: np.ndarray,
            dC_time: np.ndarray
    ):
        """
        dC_layer/dC_time = Contribution of cost gradient w.r.t. this node from node at next layer/time
        """
        pass