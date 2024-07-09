import numpy as np
from funcs import sigmoid, identity
from Node import Node

# length: int,
#             act_func: Callable[[np.ndarray], np.ndarray],
#             W_layer: np.ndarray,
#             b_layer: np.ndarray,
#             W_time: np.ndarray,
#             b_time: np.ndarray


W_layer = np.array([
    [1, 0, 5],
    [-3, 2, 0],
    [5, 1, 2]
])
b_layer = np.array([
    [3, 0, 1]
])
W_time = np.array([
    [0, 2, 0],
    [-1, 7, 3],
    [7, 0, 1]
])
b_time = np.array([
    [0, 1, 2]
])

node = Node(3, identity, None, None, None, None)
node.set_Wb(W_layer, b_layer, W_time, b_time)

h_layer = np.array([
    [1, 5, 1],
    [1, 7, 8],
    [10, 12, 9]
])
h_time = np.array([
    [3,1,2],
    [1,6,7],
    [3,2,1]
])

h_next = node.feed_forward(h_layer, h_time)
print(h_next)