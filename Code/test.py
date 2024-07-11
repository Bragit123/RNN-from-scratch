import numpy as np
from funcs import sigmoid, identity
from RNN import RNN
from InputLayer import InputLayer
from RNNLayer import RNNLayer
from OutputLayer import OutputLayer
from Node import Node

print("##### FEED FORWARD #####")

rnn = RNN()
rnn.add_InputLayer(3)
rnn.add_RNNLayer(2, sigmoid)
rnn.add_OutputLayer(3, sigmoid)
rnn.reset_weights()

np.random.seed(100)
X = np.random.uniform(size=(5,4,3))
rnn.feed_forward(X)

for i in range(rnn.n_layers):
    layer = rnn.layers[i]
    print("W_layer:")
    print(layer.W_layer)
    print()
    print("b_layer:")
    print(layer.b_layer)
    print()
    print("W_time:")
    print(layer.W_time)
    print()
    print("b_time:")
    print(layer.b_time)
    print()
    for j in range(layer.n_nodes):
        node = layer.nodes[j]
        print(f"({i},{j}):")
        print(node.get_output())


print()
print()
print()
print("##### BACKPROPAGATION #####")

node_last = rnn.layers[1].nodes[-1]
node_next_last = rnn.layers[1].nodes[-2]

dC_layer = np.random.uniform(size=(2, 5, 2))
dC_layer_last = dC_layer[0,:,:]
dC_layer_next_last = dC_layer[1,:,:]

print(f"dC_layer_last = {dC_layer_last}")

node_last.backpropagate(dC_layer_last, None)

print(f"grad_b_layer = {node_last.grad_b_layer}")
print(f"grad_b_time = {node_last.grad_b_time}")
print(f"grad_W_layer = {node_last.grad_W_layer}")
print(f"grad_W_time = {node_last.grad_W_time}")
print(f"grad_h_layer = {node_last.grad_h_layer}")
print(f"grad_h_time = {node_last.grad_h_time}")