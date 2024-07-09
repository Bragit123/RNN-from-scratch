from collections.abc import Callable
from Layer import Layer
from Node import Node
import numpy as np
 
class RNNLayer(Layer):
    def __init__(
            self,
            length: int,
            length_prev: int,
            act_func: Callable[[np.ndarray], np.ndarray],
            seed: int = 100
    ):
        """
        length = length of all the h's in this layer
        length_prev = length of all the h's in the previous layer
        seed = numpy random seed

        W_layer, b_layer, W_time, b_time:
            W = weights, b = bias
            layer = from previous layer to this one
            time = between time steps in the layer
        
        W_layer_size, b_layer_size, W_time_size, b_time_size = array shapes for the weights and biases
        """
        super().__init__(length, length_prev, act_func, seed)

        self.W_layer = None
        self.b_layer = None
        self.W_time = None
        self.b_time = None
        
        self.reset_weights()
    
    def reset_weights(self):
        """
        Reset weights and biases with normal distribution.
        The 0.01 scale for biases was found to be best when we did the CNN project, so I just kept using this here.
        """
        np.random.seed(self.seed)
        self.W_layer = np.random.normal(size=self.W_layer_size)
        self.b_layer = np.random.normal(size=self.b_layer_size) * 0.01
        self.W_time = None
        self.b_time = None
    
    def update_weights(self):
        pass

    def update_weights_all_nodes(
            self,
            new_W_layer: np.ndarray,
            new_b_layer: np.ndarray
    ):
        """
        Update the weights and biases in all nodes of the layer.
        """
        new_W_time = None
        new_b_time = None
        for node in self.nodes:
            node.set_Wb(new_W_layer, new_b_layer, new_W_time, new_b_time)
    
    def add_node(self):
        """
        Add a node with the weights and biases specified by layer
        """
        new_node = Node(self.length, self.act_func, self.W_layer, self.b_layer, self.W_time, self.b_time)
        self.nodes.append(new_node)
        self.n_nodes += 1
    
    def remove_nodes(self):
        """
        Remove all the nodes created for this layer.
        NOTE: The weights and biases of the nodes are still stored in the layer, so we can easily
            create new nodes. Removing the nodes is used to allow the input length to vary with
            each call of feed_forward().
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
        self.remove_nodes()
        n_nodes_prev = prev_layer.n_nodes

        for i in range(n_nodes_prev):
            # Get output of node from previous layer
            prev_layer_node = prev_layer.nodes[i]
            h_layer = prev_layer_node.get_output()
            
            # Create and compute new node at this time step
            self.add_node()
            new_node = self.nodes[i]

            # No info transfer between time steps in output layer
            output = new_node.feed_forward(h_layer, None)
            
    
    def backpropagate(
            self,
            dC: np.ndarray
    ):
        pass