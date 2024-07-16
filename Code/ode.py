import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from sklearn.preprocessing import MinMaxScaler
from RNN_module.RNN import RNN
from RNN_module.funcs import identity, CostOLS
from RNN_module.schedulers import Adam

## Generate data
x0 = 1
x1 = 10
N = 20
x = np.linspace(x0, x1, N)

y0 = 4
alpha = -0.1
t_orig = y0 * np.exp(-alpha*x)

X = x[np.newaxis,:,np.newaxis]
t = t_orig[np.newaxis,:,np.newaxis]

## Set parameters
cost_func = CostOLS
act_func_hidden = identity
act_func_output = identity
eta = 0.001
scheduler = Adam(eta, 0.9, 0.999)

## Create RNN
rnn = RNN(cost_func, scheduler)
rnn.add_InputLayer(1)
rnn.add_RNNLayer(2, act_func_hidden)
rnn.add_OutputLayer(1, act_func_output)
rnn.reset_weights()

## Train network
epochs = 200
batches = 1
lmbd = 0.001
scores = rnn.train(X, t, epochs=epochs, batches=batches, lmbd=lmbd, store_output=True)

## Extract output
epoch_arr = np.arange(1, epochs+1)
train_error = scores["train_error"]
y_history = scores["y_train_history"]

## Plot error during training
plt.figure()
plt.plot(epoch_arr, train_error, label="Training error")
plt.yscale("log")
plt.legend()
plt.savefig("Figures/ode_error.pdf")

## Plot result compared to target
x = X[0,:,0]
y = y_history[-1,0,:,0]
plt.figure()
plt.plot(x, t_orig, "k", label="Target")
plt.plot(x, y, "b--", label="Output")
plt.legend()
plt.savefig("Figures/ode.pdf")