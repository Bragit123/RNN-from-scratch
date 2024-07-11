import numpy as np
from funcs import sigmoid, identity, CostOLS
from RNN import RNN
from schedulers import Constant, Adam
from InputLayer import InputLayer
from RNNLayer import RNNLayer
from OutputLayer import OutputLayer
from Node import Node

### Feed forward

cost_func = CostOLS
act_func = sigmoid
scheduler = Adam(eta=0.1, rho=0.9, rho2=0.999)

rnn = RNN(cost_func)
rnn.add_InputLayer(3)
rnn.add_RNNLayer(2, act_func, scheduler)
rnn.add_OutputLayer(3, act_func, scheduler)
rnn.reset_weights()

np.random.seed(100)
X = np.random.uniform(size=(5,4,3))
output = rnn.feed_forward(X)

### Backpropagation

output_shape = output.shape
target = np.random.uniform(size=output_shape)

print(cost_func(target)(output))
for i in range(100):
    rnn.backpropagate(output, target, lmbd=0.01)
    output = rnn.feed_forward(X)
    print(cost_func(target)(output))