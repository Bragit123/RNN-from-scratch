from collections.abc import Callable
from schedulers import Scheduler
from Layer import Layer
from InputLayer import InputLayer
from OutputLayer import OutputLayer
from RNNLayer import RNNLayer
from Node import Node
import numpy as np

class RNN:
    def __init__(
            self,
            seed: int = 100
    ):
        self.layers: list[Layer] = []
        self.n_layers = 0
        self.seed = seed
    
    def reset_weights(self):
        for layer in self.layers:
            layer.reset_weights()

    def feed_forward(
            self,
            X: np.ndarray
    ):
        self.layers[0].feed_forward(X)
        for i in range(1, self.n_layers):
            layer = self.layers[i]
            prev_layer = self.layers[i-1]
            layer.feed_forward(prev_layer)

    def train(
            self,
            X_train: np.ndarray,
            y_train: np.ndarray,
            X_test: np.ndarray,
            y_test: np.ndarray,
            lmbd: float,
            epochs: int
    ):
        pass

    def add_InputLayer(
            self,
            n_features: int
    ):
        """
        n_features = number of features expected from the input.
        """
        layer = InputLayer(n_features)
        self._add_layer(layer)
    
    def add_RNNLayer(
            self,
            n_features: int,
            act_func: Callable[[np.ndarray], np.ndarray],
            scheduler: Scheduler
    ):
        """
        n_features = number of features in this layer
        n_features_prev = number of features in the previous layer
        act_func = activation function for this layer
        seed = numpy random seed
        """
        prev_layer = self.layers[-1]
        n_features_prev = prev_layer.n_features
        layer = RNNLayer(n_features, n_features_prev, act_func, scheduler, self.seed)
        self._add_layer(layer)

    def add_OutputLayer(
            self,
            n_features: int,
            act_func: Callable[[np.ndarray], np.ndarray]
    ):
        """
        n_features = number of features in this layer
        n_features_prev = number of features in the previous layer
        act_func = activation function for this layer
        seed = numpy random seed
        """
        prev_layer = self.layers[-1]
        n_features_prev = prev_layer.n_features
        layer = OutputLayer(n_features, n_features_prev, act_func, self.seed)
        self._add_layer(layer)
    
    def _add_layer(
            self,
            layer: Layer
    ):
        self.layers.append(layer)
        self.n_layers += 1