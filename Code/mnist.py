import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import datasets
from tensorflow.keras.utils import to_categorical # This allows using categorical cross entropy as the cost function
from RNN_module.funcs import CostLogReg, identity, sigmoid, RELU, LRELU, tanh, softmax
from RNN_module.schedulers import Adam
from RNN_module.RNN import RNN


# MNIST consists of 60 000 samples. data_frac decides the portion of these we will use
data_frac = 0.01

## Retrieve and normalize the data (split into train and test)
digits = datasets.mnist.load_data(path="mnist.npz")
(X_train, t_train), (X_val, t_val) = digits
X_train, X_val = X_train/255.0, X_val/255.0

# Retrieve only a subset of the images (how many is decided by data_frac)
X_train = X_train[:][:][0:int(data_frac*len(X_train[:][:]))]
X_val = X_val[:][:][0:int(data_frac*len(X_val[:][:]))]
t_train = t_train[0:int(data_frac*len(t_train))]
t_val = t_val[0:int(data_frac*len(t_val))]

#Transforming the labels from a single digit to an array of length 10 with the digit corresponding to the index
t_train = to_categorical(t_train)
t_val = to_categorical(t_val)

## Plot some images for visual understanding
n_rows = 5
n_cols = 5
fig, axs = plt.subplots(n_rows, n_cols, constrained_layout=True)
for row in range(n_rows):
    for col in range(n_cols):
        ax = axs[row,col]
        ax.set_axis_off()
        ind = n_cols * row + col
        label = np.argmax(t_train[ind])
        ax.set_title(label, loc="left")
        ax.imshow(X_train[ind])
fig.savefig("Figures/MNIST/mnist_sample_grid.pdf")


## Set RNN parameters
eta = 0.01
lmbd = 0.001
seed = 100

act_func_hidden = tanh
act_func_output = softmax
cost_func = CostLogReg
scheduler = Adam(eta, 0.9, 0.999)

n_features_input = X_train.shape[2]
n_features_hidden = 100
n_features_output = t_train.shape[1]

## Create RNN
rnn = RNN(cost_func, scheduler, seed)
rnn.add_InputLayer(n_features_input)
rnn.add_RNNLayer(n_features_hidden, act_func_hidden)
rnn.add_SingleOutputLayer(n_features_output, act_func_output)
rnn.reset_weights()

## Train network
epochs = 10
batches = 10
scores = rnn.train(X_train, t_train, X_val, t_val, epochs, batches, lmbd)

## Extract output
epoch_arr = np.arange(1, epochs+1)
train_error = scores["train_error"]
val_error = scores["val_error"]
train_accuracy = scores["train_accuracy"]
val_accuracy = scores["val_accuracy"]

## Plot error during training
plt.figure()
plt.plot(epoch_arr, train_error, label="Training error")
plt.plot(epoch_arr, val_error, label="Validation error")
plt.legend()
plt.savefig("Figures/MNIST/mnist_error.pdf")

## Plot accuracy during training
plt.figure()
plt.plot(epoch_arr, train_accuracy, label="Training accuracy")
plt.plot(epoch_arr, val_accuracy, label="Validation accuracy")
plt.legend()
plt.savefig("Figures/MNIST/mnist_accuracy.pdf")

y_train = rnn.feed_forward(X_train)
print()
for i in range(10):
    print(f"{np.max(y_train[i]):.2f} : {np.sum(y_train[i]):.2f}")

## Plot some images for visual understanding
pred = rnn.predict(X_val)
n_rows = 5
n_cols = 5
fig, axs = plt.subplots(n_rows, n_cols, constrained_layout=True)
for row in range(n_rows):
    for col in range(n_cols):
        ax = axs[row,col]
        ax.set_axis_off()
        ind = n_cols * row + col
        label = np.argmax(pred[ind])
        ax.set_title(label, loc="left")
        ax.imshow(X_val[ind])
fig.savefig("Figures/MNIST/mnist_sample_grid_output.pdf")