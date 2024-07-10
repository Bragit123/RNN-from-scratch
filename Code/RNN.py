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
        self.layers = []
        self.seed = seed
    
    def reset_weights(self):
        pass

    def feed_forward(
            self,
            X: np.ndarray
    ):
        pass

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

    def add_RNNLayer(
            self,
            W_layer: np.ndarray,
            b_layer: np.ndarray,
            W_time: np.ndarray,
            b_time: np.ndarray,
    ):
        pass