from collections.abc import Callable
from copy import copy
from jax import vmap
from .funcs import grad_softmax, derivate
from .schedulers import Scheduler
from .Layer import Layer
from .Node import Node
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
        super().__init__(n_features, seed)

        self.n_features_prev = n_features_prev
        self.act_func = act_func
        
        self.W_layer_size = (self.n_features_prev, self.n_features)
        self.b_layer_size = (1, self.n_features)

        self.nodes = []
        self.n_nodes = 0

        self.W_layer = None
        self.b_layer = None

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
        self.nodes[0].set_Wb(self.W_layer, self.b_layer)
    
    def add_node(self):
        """
        Dense layer has no nodes. This method does nothing.
        """
        new_node = Node(self.n_features, self.act_func, self.W_layer, self.b_layer)
        self.nodes.append(new_node)
        self.n_nodes += 1
    
    def remove_nodes(self):
        """
        Dense layer has no nodes. This method does nothing.
        """
        self.nodes = []
        self.n_nodes = 0

    def feed_forward(
            self,
            prev_layer: Layer
    ):
        """
        Compute the output of this layer from the input (the output from the previous layer), and
        feed forward this to the next layer.
        """
        ## Get output from last node of previous layer
        prev_node = prev_layer.nodes[-1]
        h_layer = prev_node.h_output

        self.add_node()
        new_node = self.nodes[0]
        output = new_node.feed_forward(h_layer)

        return output
            
    
    def backpropagate(
            self,
            next_layer_or_dC: Layer | np.ndarray,
            lmbd: float = 0.01
    ):
        """
        next_layer = next layer, if this is not the last
        dC = Gradient of the cost function for the specific target
        dC_shape = (batch_size, n_features)
        NOTE: Should input either next_layer or dC, not both. Use dC if this is the last layer, and next_layer if not.
        """
        ## Get dC from next_layer_or_dC.
        if isinstance(next_layer_or_dC, Layer):
            # If next_layer_or_dC is a Layer, extract dC from last node.
            dC = next_layer_or_dC.nodes[0].grad_h_layer
        else:
            # If next_layer_or_dC is not a layer, it is the cost gradient.
            dC = next_layer_or_dC
        
        ## Backpropagate through the node
        node = self.nodes[0]
        node.backpropagate(dC_layer=dC, dC_time=None, lmbd=lmbd)

        ## Update weights and biases
        grad_W_layer = node.grad_W_layer
        grad_b_layer = node.grad_b_layer

        self.W_layer -= self.scheduler_W_layer.update_change(grad_W_layer)
        self.b_layer -= self.scheduler_b_layer.update_change(grad_b_layer)

        self.update_weights_all_nodes()