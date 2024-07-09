from collections.abc import Callable
from funcs import identity
import numpy as np

class Node:
    def __init__(
            self,
            length: int,
            act_func: Callable[[np.ndarray], np.ndarray] = identity,
            W_layer: np.ndarray = None,
            b_layer: np.ndarray = None,
            W_time: np.ndarray = None,
            b_time: np.ndarray = None
    ):
        """
        length = length of vector h at this node
        W = weights, b = bias
        _layer = from previous layer to this one
        _time = from previous time step to this one
        output = output from feed_forward through this node
        """
        self.length = length
        self.act_func = act_func
        self.W_layer = W_layer
        self.b_layer = b_layer
        self.W_time = W_time
        self.b_time = b_time

        self.output = None
    
    def set_Wb(
            self,
            W_layer: np.ndarray,
            b_layer: np.ndarray,
            W_time: np.ndarray,
            b_time: np.ndarray
    ):
        self.W_layer = W_layer
        self.b_layer = b_layer
        self.W_time = W_time
        self.b_time = b_time
    
    def get_output(self):
        return self.output
    
    def set_output(self, output: np.ndarray):
        self.output = output

    def feed_forward(
            self,
            h_layer: np.ndarray,
            h_time: np.ndarray = None
    ):
        """
        h_layer/h_time: Output from node at previous layer/time
        first_node: True if this is the first node of the layer.

        h_shape = (n_batch, input_length)
        """
        num_inputs = h_layer.shape[0]

        ## Compute weighted sum z for this node.
        z_layer = h_layer @ self.W_layer + self.b_layer # Dimension example: (100,5)@(5,7) + (7) = (100,7)

        if h_time is None:
            # This node is at the first time step, thus not receiving any input from previous time steps.
            z_time = np.zeros((num_inputs, self.length))
        else:
            z_time = h_time @ self.W_time + self.b_time
        
        z_output = z_layer + z_time

        ## Compute activation of the node
        h_output = self.act_func(z_output)

        self.set_output(h_output) # Save the output in the node
        return h_output # Return output
        

    def backpropagate(
            self,
            dC_layer: np.ndarray,
            dC_time: np.ndarray,
            last_node: bool
    ):
        """
        dC_layer/dC_time = Contribution of cost gradient w.r.t. this node from node at next layer/time
        last_node: True if this is the last node of the layer.
        """
        pass