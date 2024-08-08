import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam as tf_Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import to_categorical # This allows using categorical cross entropy as the cost function

# Load MNIST dataset
(X_train, y_train), (X_val, y_val) = mnist.load_data()

# Extract a subset of the data
n_train = X_train.shape[0]
n_val = X_val.shape[0]
data_frac = 0.01
train_end = int(n_train*data_frac)
val_end = int(n_val*data_frac)
X_train = X_train[:train_end]
X_val = X_val[:val_end]
y_train = y_train[:train_end]
y_val = y_val[:val_end]

y_train = to_categorical(y_train)
y_val = to_categorical(y_val)

# Normalize data
X_train = X_train.astype('float32') / 255.0
X_val = X_val.astype('float32') / 255.0

# Parameters BEST SO FAR
eta = 0.001
lmbd = 0.001
n_features_hidden = 300
n_dense_features = 10
epochs = 10
batches = 60

n_features_output = 10
batch_size = int(np.ceil(X_train.shape[0]/batches))

# Build RNN model
model = Sequential()
model.add(SimpleRNN(n_features_hidden, activation='relu',
                input_shape=(X_train.shape[1], X_train.shape[2]),
                kernel_regularizer=l2(lmbd)))
model.add(Dense(10, activation='softmax', kernel_regularizer=l2(lmbd)))

# Compile model
optimizer = tf_Adam(learning_rate=eta)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Train model
history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
                    validation_data=(X_val, y_val))

# Evaluate model
val_loss, val_accuracy = model.evaluate(X_val, y_val)
print(f'Val accuracy: {val_accuracy}, Val loss: {val_loss}')

# Training error history
training_accuracy_history = history.history["accuracy"]
val_accuracy_history = history.history["val_accuracy"]
training_loss_history = history.history['loss']
val_loss_history = history.history['val_loss']

print("Training loss history:", training_loss_history)
print("Validation loss history:", val_loss_history)





from RNN_module.RNN import RNN
from RNN_module.funcs import CostLogReg, RELU, softmax
from RNN_module.schedulers import Adam

n_featuers_input = X_train.shape[2]
act_func_hidden = RELU
act_func_output = softmax
cost_func = CostLogReg
scheduler = Adam(eta, 0.9, 0.999)

rnn = RNN(cost_func, scheduler)
rnn.add_InputLayer(n_featuers_input)
rnn.add_RNNLayer(n_features_hidden, act_func_hidden)
rnn.add_DenseLayer(n_features_output, act_func_output, is_last_layer=True)

scores = rnn.train(X_train, y_train, X_val, y_val, epochs, batches, lmbd, store_output=True)

epoch_arr = np.arange(1, epochs+1)
train_error = scores["train_error"]
val_error = scores["val_error"]
train_accuracy = scores["train_accuracy"]
val_accuracy = scores["val_accuracy"]

## Plot error during training
plt.figure()
plt.plot(epoch_arr, train_error, label="Training error")
plt.plot(epoch_arr, val_error, label="Validation error")
plt.plot(epoch_arr, training_loss_history, label="(TF) Training error")
plt.plot(epoch_arr, val_loss_history, label="(TF) Validation error")
plt.legend()
plt.savefig("mnist_error.pdf")

## Plot accuracy during training
plt.figure()
plt.plot(epoch_arr, train_accuracy, label="Training accuracy")
plt.plot(epoch_arr, val_accuracy, label="Validation accuracy")
plt.plot(epoch_arr, training_accuracy_history, label="(TF) Training accuracy")
plt.plot(epoch_arr, val_accuracy_history, label="(TF) Validation accuracy")
plt.legend()
plt.savefig("mnist_accs.pdf")