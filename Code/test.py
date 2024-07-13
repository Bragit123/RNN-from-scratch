import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from funcs import sigmoid, identity, tanh, RELU, CostOLS
from sklearn.model_selection import train_test_split
from RNN import RNN
from schedulers import Adam

cost_func = CostOLS
act_func_hidden = identity
# act_func_hidden = tanh
# act_func_hidden = RELU
act_func_output = identity
scheduler = Adam(eta=0.001, rho=0.9, rho2=0.999)

## Create RNN
rnn = RNN(cost_func)
rnn.add_InputLayer(1)
rnn.add_RNNLayer(1, act_func_hidden, scheduler)
rnn.add_OutputLayer(1, act_func_output, scheduler)
rnn.reset_weights()

## Retrieve data
N = 100
weather_data = pd.read_csv("../Data/clean_weather.csv")
x = weather_data["tmax"][:N].to_numpy()
t = weather_data["tmax_tomorrow"][:N].to_numpy()

# Give data correct dimensions
x = x[np.newaxis, :, np.newaxis]
t = t[np.newaxis, :, np.newaxis]

## Train network on data
epochs = 50
batches = 1
scores = rnn.train(x, t, epochs=epochs, batches=batches, lmbd = 0.001, store_output=True)

## Extract output
epoch_arr = np.arange(1, epochs+1)
y_history = scores["y_train_history"]
train_error = scores["train_error"]

plt.figure()
plt.plot(epoch_arr, train_error)
plt.yscale("log")
plt.savefig("weather_error.pdf")

seq_ind = np.arange(N)
y = y_history[-1,0,:,0]
plt.figure()
plt.plot(seq_ind, t[0,:,0], label="Target")
plt.plot(seq_ind, y, ".", label="Output")
plt.legend()
plt.savefig("weather.pdf")

## Create animation
