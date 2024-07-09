import numpy as np
from funcs import sigmoid, identity
from InputLayer import InputLayer
from RNNLayer import RNNLayer
from Node import Node

# # length: int,
# #             act_func: Callable[[np.ndarray], np.ndarray],
# #             W_layer: np.ndarray,
# #             b_layer: np.ndarray,
# #             W_time: np.ndarray,
# #             b_time: np.ndarray


# W_layer = np.array([
#     [1, 0, 5],
#     [-3, 2, 0],
#     [5, 1, 2]
# ])
# b_layer = np.array([
#     [3, 0, 1]
# ])
# W_time = np.array([
#     [0, 2, 0],
#     [-1, 7, 3],
#     [7, 0, 1]
# ])
# b_time = np.array([
#     [0, 1, 2]
# ])

# node = Node(3, identity, None, None, None, None)
# node.set_Wb(W_layer, b_layer, W_time, b_time)

# h_layer = np.array([
#     [1, 5, 1],
#     [1, 7, 8],
#     [10, 12, 9]
# ])
# h_time = np.array([
#     [3,1,2],
#     [1,6,7],
#     [3,2,1]
# ])

# h_next = node.feed_forward(h_layer, h_time)
# print(h_next)


##################


input_layer = RNNLayer(3, 3, identity)
input_layer.reset_weights()
input_node1 = Node(3, identity, input_layer.W_layer,input_layer.b_layer, input_layer.W_time, input_layer.b_time)
input_node2 = Node(3, identity, input_layer.W_layer,input_layer.b_layer, input_layer.W_time, input_layer.b_time)
input_node3 = Node(3, identity, input_layer.W_layer,input_layer.b_layer, input_layer.W_time, input_layer.b_time)
input_node1.output = np.random.uniform(size=(5,3))
input_node2.output = np.random.uniform(size=(5,3))
input_node3.output = np.random.uniform(size=(5,3))
input_layer.nodes = [input_node1, input_node2, input_node3]
input_layer.n_nodes = 3 

X = np.random.uniform(size=(5,4,3))
input_layer = InputLayer(X)

hidden_layer = RNNLayer(2, 3, identity)
hidden_layer.reset_weights()

hidden_layer.feed_forward(input_layer)

print("Input:")
print(f"   x1 =")
print(input_layer.nodes[0].get_output())
print()
print(f"   x2 =")
print(input_layer.nodes[1].get_output())
print()
print(f"   x3 =")
print(input_layer.nodes[2].get_output())
print()
print(f"   x4 =")
print(input_layer.nodes[3].get_output())

print()
print()
print()

print("Hidden:")
print(f"   W_layer =")
print(hidden_layer.W_layer)
print()
print(f"   b_layer =")
print(hidden_layer.b_layer)
print()
print(f"   W_time =")
print(hidden_layer.W_time)
print()
print(f"   b_time =")
print(hidden_layer.b_time)
print()
print(f"   h1 =")
print(hidden_layer.nodes[0].get_output())
print()
print(f"   h2 =")
print(hidden_layer.nodes[1].get_output())
print()
print(f"   h3 =")
print(hidden_layer.nodes[2].get_output())
print()
print(f"   h4 =")
print(hidden_layer.nodes[3].get_output())
print()
print(f"   n_nodes = {hidden_layer.n_nodes}")