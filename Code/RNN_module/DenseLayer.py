from collections.abc import Callable
from copy import copy
from jax import vmap
from .funcs import grad_softmax, derivate
from .schedulers import Scheduler
from .Layer import Layer
import numpy as np
 
class DenseLayer(Layer):
    def __init__(
            self,
            n_features: int,
            n_features_prev: int,
            act_func: Callable[[np.ndarray], np.ndarray],
            scheduler: Scheduler,
            seed: int = 100
    ):
        """
        n_features = number of features in this layer
        n_features_prev = number of features in the previous layer
        act_func = activation function for this layer
        seed = numpy random seed

        W_layer, b_layer, W_time, b_time:
            W = weights, b = bias
            layer = from previous layer to this one
            time = between time steps in the layer
        
        W_layer_size, b_layer_size, W_time_size, b_time_size = array shapes for the weights and biases
        """
        super().__init__(n_features, n_features_prev, act_func, seed)
        self.weights_size = (n_features_prev, n_features)
        self.bias_length = n_features

        self.weights = None
        self.bias = None

        self.h_prev = None
        self.z_output = None
        self.h_output = None

        self.grad_weights = None
        self.grad_bias = None
        self.grad_h_prev = None

        self.scheduler_weights = copy(scheduler)
        self.scheduler_bias = copy(scheduler)

        self.is_dense = True
        
        self.reset_weights()
    
    def reset_weights(self):
        """
        Reset weights and biases with normal distribution.
        The 0.01 scale for biases was found to be best when we did the CNN project, so I just kept using this here.
        """
        np.random.seed(self.seed)
        self.weights = np.random.normal(size=self.weights_size)
        self.bias = np.random.normal(size=(1, self.bias_length)) * 0.01
    
    def reset_schedulers(self):
        """
        Reset the schedulers of the layer.
        """
        self.scheduler_weights.reset()
        self.scheduler_bias.reset()

    def update_weights_all_nodes(self):
        """
        Dense layer has no nodes. This method does nothing.
        """
        return
    
    def add_node(self):
        """
        Dense layer has no nodes. This method does nothing.
        """
        return
    
    def remove_nodes(self):
        """
        Dense layer has no nodes. This method does nothing.
        """
        return

    def feed_forward(
            self,
            prev_layer: Layer
    ):
        """
        Compute the output of this layer from the input (the output from the previous layer), and
        feed forward this to the next layer.
        """
        ## Get output from previous layer (last node if RNNLayer)
        # if isinstance(prev_layer, DenseLayer):
        if prev_layer.is_dense:
            self.h_prev = prev_layer.h_output
        else:
            last_node = prev_layer.nodes[-1]
            self.h_prev = last_node.h_output
        
        self.z_output = self.h_prev @ self.weights + self.bias
        self.h_output = self.act_func(self.z_output)

        return self.h_output
            
    
    def backpropagate(
            self,
            next_layer: Layer = None,
            dC: np.ndarray = None,
            lmbd: float = 0.01
    ):
        """
        next_layer = next layer, if this is not the last
        dC = Gradient of the cost function for the specific target
        dC_shape = (batch_size, n_features)
        NOTE: Should input either next_layer or dC, not both. Use dC if this is the last layer, and next_layer if not.
        """
        ## Get dC from next layer if it is given.
        if next_layer is not None:
            dC = next_layer.grad_h_prev

        ## Find gradient of activation function
        if self.act_func.__name__ == "softmax":
            grad_act = grad_softmax(self.z_output)
        else:
            grad_act = vmap(vmap(derivate(self.act_func)))(self.z_output)
        
        ## Compute delta
        n_batches = self.z_output.shape[0]
        delta = dC * grad_act # Hadamard product (elementwise)
        
        ## Compute gradients
        grad_weights = self.h_prev.T @ delta / n_batches
        grad_weights = grad_weights + self.weights * lmbd # Add regularization
        grad_bias = np.sum(delta, axis=0).reshape(1, delta.shape[1]) / n_batches
        self.grad_h_prev = delta @ self.weights.T

        ## Update weights and bias
        self.weights -= self.scheduler_weights.update_change(grad_weights)
        self.bias -= self.scheduler_bias.update_change(grad_bias)