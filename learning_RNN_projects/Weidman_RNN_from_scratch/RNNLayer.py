import numpy as np
from typing import List

class Layer:
    def __init__(self):
        pass

    def __forward__(self):
        pass

class RNNLayer(Layer):
    def __init__(self, layers: type[Layer]):
        self.layers = layers

    def forward(self, x_batch: np.ndarray) -> np.ndarray:
        x_out = x_batch
        for layer in self.layers:
            x_out = layer.forward(x_out)
        
        return x_out