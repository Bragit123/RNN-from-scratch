import numpy as np
from funcs import sigmoid, identity
from RNN import RNN
from schedulers import Constant, Adam
from InputLayer import InputLayer
from RNNLayer import RNNLayer
from OutputLayer import OutputLayer
from Node import Node

print("##### FEED FORWARD #####")

act_func = sigmoid
# scheduler = Constant(eta=0.1)
scheduler = Adam(eta=0.1, rho=0.9, rho2=0.999)

rnn = RNN()
rnn.add_InputLayer(3)
rnn.add_RNNLayer(2, act_func, scheduler)
rnn.add_OutputLayer(3, act_func)
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

last_layer = rnn.layers[-1]
grad_layer = np.random.uniform(size=(last_layer.n_nodes, 5, 2))
for i in range(last_layer.n_nodes):
    node = last_layer.nodes[i]
    node.grad_h_layer = grad_layer[i,:,:]

print(f"grad_layer = {grad_layer}")

hidden_layer = rnn.layers[1]
hidden_layer.backpropagate(last_layer, lmbd=0.01)

for i in range(hidden_layer.n_nodes):
    node = hidden_layer.nodes[i]
    print(f"i = {i}")
    print(f"grad_b_layer = {node.grad_b_layer}")
    print(f"grad_b_time = {node.grad_b_time}")
    print(f"grad_W_layer = {node.grad_W_layer}")
    print(f"grad_W_time = {node.grad_W_time}")
    print(f"grad_h_layer = {node.grad_h_layer}")
    print(f"grad_h_time = {node.grad_h_time}")
    print()