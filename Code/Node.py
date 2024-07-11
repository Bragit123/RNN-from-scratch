from collections.abc import Callable
from funcs import identity, derivate
from jax import vmap
import numpy as np

class Node:
    def __init__(
            self,
            n_features: int,
            act_func: Callable[[np.ndarray], np.ndarray] = identity,
            W_layer: np.ndarray = None,
            b_layer: np.ndarray = None,
            W_time: np.ndarray = None,
            b_time: np.ndarray = None
    ):
        """
        n_features = number of features for this node
        W = weights, b = bias
        _layer = from previous layer to this one
        _time = from previous time step to this one
        output = output from feed_forward through this node
        """
        self.n_features = n_features
        self.act_func = act_func
        self.W_layer = W_layer
        self.b_layer = b_layer
        self.W_time = W_time
        self.b_time = b_time

        ## Values from feed_forward()
        self.h_layer = None # h from previous layer
        self.h_time = None # h from previous time step
        self.z_output = None # z for this node
        self.h_output = None # h from this node

        ## Values from backpropagate()
        self.grad_b_layer = None
        self.grad_b_time = None
        self.grad_W_layer = None
        self.grad_W_time = None
        self.grad_h_layer = None
        self.grad_h_time = None
    
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
        return self.h_output
    
    def set_output(self, output: np.ndarray):
        self.h_output = output

    def feed_forward(
            self,
            h_layer: np.ndarray,
            h_time: np.ndarray = None
    ):
        """
        h_layer/h_time: Output from node at previous layer/time
        h_shape = (n_batches, n_features)
        """
        ## Save h_layer and h_time for use in backpropagation
        self.h_layer = h_layer
        self.h_time = h_time

        num_inputs = h_layer.shape[0]

        ## Compute weighted sum z for this node.
        z_layer = h_layer @ self.W_layer + self.b_layer # Dimension example: (100,5)@(5,7) + (7) = (100,7)

        if h_time is None:
            # This node is at the first time step, thus not receiving any input from previous time steps.
            z_time = np.zeros((num_inputs, self.n_features))
        else:
            z_time = h_time @ self.W_time + self.b_time
        
        self.z_output = z_layer + z_time # Save the weighted sum in the node

        ## Compute activation of the node
        h_output = self.act_func(self.z_output)

        self.h_output = h_output # Save the output in the node
        return h_output # Return output
        

    def backpropagate(
            self,
            dC_layer: np.ndarray,
            dC_time: np.ndarray = None,
            lmbd: float = 0.01
    ):
        """
        dC_layer/dC_time = Contribution of cost gradient w.r.t. this node from node at next layer/time
        dC_shape = (n_batches, n_features), i.e., the same as h_shape in feed_forward
        """
        n_batches = self.h_output.shape[0]
        
        ## Total gradient is the sum of the gradient from "next" layer and time
        if dC_time is None:
            # If this is the last node in the layer, the gradient is just the gradient from the next layer
            dC = dC_layer
        else:
            dC = dC_layer + dC_time
        
        ## delta (gradient of cost w.r.t. z)
        grad_act = vmap(vmap(derivate(self.act_func)))(self.z_output) # vmap is necessary for jax to vectorize gradient properly
        delta = grad_act * dC # Hadamard product, i.e., elementwise multiplication

        ## Gradients w.r.t. bias
        self.grad_b_layer = np.sum(delta, axis=0) / n_batches
        self.grad_b_time = np.sum(delta, axis=0) / n_batches

        ## Gradients w.r.t. weights
        # Need to transpose h and not delta in order for matrices to match up correctly, since we have batches along rows, and features along columns
        self.grad_W_layer = self.h_layer.T @ delta / n_batches
        self.grad_W_layer = self.grad_W_layer + self.W_layer * lmbd # Regularization factor
        self.grad_W_time = self.h_time.T @ delta / n_batches
        self.grad_W_time = self.grad_W_time + self.W_time * lmbd # Regularization factor

        ## Gradients w.r.t. input from previous nodes
        # Need to not transpose delta in order for matrices to match up correctly, since we have batches along rows, and features along columns
        self.grad_h_layer = delta @ self.W_layer.T
        self.grad_h_time = delta @ self.W_time.T

