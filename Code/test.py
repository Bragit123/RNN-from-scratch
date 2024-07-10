import numpy as np
from funcs import sigmoid, identity
from InputLayer import InputLayer
from OutputLayer import OutputLayer
from RNNLayer import RNNLayer
from Node import Node

np.random.seed(100)
X = np.random.uniform(size=(4,10,3))
input_layer = InputLayer(3)
input_layer.feed_forward(X)

hidden_layer = RNNLayer(2, 3, identity)
hidden_layer.reset_weights()
hidden_layer.feed_forward(input_layer)

output_layer = OutputLayer(3, 2, identity)
output_layer.reset_weights()
output_layer.feed_forward(hidden_layer)

print("Input:")
for i in range(input_layer.n_nodes):
    print(f"   x[{i}] =")
    print(input_layer.nodes[i].get_output())
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
for i in range(hidden_layer.n_nodes):
    print(f"   h[{i}] =")
    print(hidden_layer.nodes[i].get_output())
    print()

print()
print()

print("Output:")
print(f"   W_layer =")
print(output_layer.W_layer)
print()
print(f"   b_layer =")
print(output_layer.b_layer)
print()
print(f"   W_time =")
print(output_layer.W_time)
print()
print(f"   b_time =")
print(output_layer.b_time)
print()
for i in range(output_layer.n_nodes):
    print(f"   o[{i}] =")
    print(output_layer.nodes[i].get_output())
