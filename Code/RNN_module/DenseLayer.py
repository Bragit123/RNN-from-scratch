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

        ##############################
        # self.nodes = []
        # self.n_features = n_features
        # self.n_features_prev = n_features_prev
        # self.act_func = act_func
        # self.seed = seed
        
        # self.W_layer_size = (self.n_features_prev, self.n_features)
        # self.b_layer_size = (1, self.n_features)
        # self.W_time_size = (self.n_features, self.n_features)
        # self.b_time_size = (1, self.n_features)

        # self.n_nodes = 0
        ##############################
        
        self.W_layer_size = (n_features_prev, n_features)
        self.b_layer_size = (1, n_features)

        self.W_layer = None
        self.b_layer = None

        self.h_prev = None
        self.z_output = None
        self.h_output = None

        self.grad_W_layer = None
        self.grad_b_layer = None
        self.grad_h_prev = None

        self.scheduler_W_layer = copy(scheduler)
        self.scheduler_b_layer = copy(scheduler)

        self.is_dense = True
        
        self.reset_weights()
    
    def reset_weights(self):
        """
        Reset weights and biases with normal distribution.
        The 0.01 scale for biases was found to be best when we did the CNN project, so I just kept using this here.
        """
        np.random.seed(self.seed)
        self.W_layer = np.random.normal(size=self.W_layer_size)
        self.b_layer = np.random.normal(size=self.b_layer_size) * 0.01
    
    def reset_schedulers(self):
        """
        Reset the schedulers of the layer.
        """
        self.scheduler_W_layer.reset()
        self.scheduler_b_layer.reset()

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
        
        self.z_output = self.h_prev @ self.W_layer + self.b_layer
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
        grad_W_layer = self.h_prev.T @ delta / n_batches
        grad_W_layer = grad_W_layer + self.W_layer * lmbd # Add regularization
        grad_b_layer = np.sum(delta, axis=0).reshape(1, delta.shape[1]) / n_batches
        self.grad_h_prev = delta @ self.W_layer.T

        ## Update weights and bias
        self.W_layer -= self.scheduler_W_layer.update_change(grad_W_layer)
        self.b_layer -= self.scheduler_b_layer.update_change(grad_b_layer)