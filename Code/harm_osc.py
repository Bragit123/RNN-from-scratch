import numpy as np
import matplotlib.pyplot as plt

# Parameters for the damped harmonic oscillator
m = 1.0  # mass
c = 0.1  # damping coefficient
k = 1.0  # spring constant
# F = lambda t: np.sin(t)  # forcing function
F = lambda t: 0  # forcing function

# Time settings
t_max = 30
dt = 0.1
t = np.arange(0, t_max, dt)

# Initialize displacement and velocity
x = np.zeros_like(t)
v = np.zeros_like(t)
x[0], v[0] = 1, 0  # initial conditions

# Simulate the damped harmonic oscillator
for i in range(1, len(t)):
    a = (F(t[i-1]) - c * v[i-1] - k * x[i-1]) / m
    v[i] = v[i-1] + a * dt
    x[i] = x[i-1] + v[i] * dt

# Plot the generated data
plt.figure()
plt.plot(t, x, label='Displacement x(t)')
plt.xlabel('Time t')
plt.ylabel('Displacement x')
plt.legend()
plt.savefig("Figures/HARM_OSC/euler_harm.pdf")

X = x[:-1]
y = x[1:]

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2


## Some parameters

# # Good for my RNN
# eta = 0.001
# lam = 0.01
# n_nodes_hidden = 5

# Good for my RNN (no forcing term)
eta = 0.001
lam = 0.0001
n_nodes_hidden = 3

# # Good for Tensorflow
# eta = 0.1
# lam = 0.001
# n_nodes_hidden = 20

# General
epochs = 10
batch_size = 1
batches = 1
seq_length = X.shape[0]

# Build the RNN model
model = Sequential([
    SimpleRNN(units=n_nodes_hidden, activation='tanh', input_shape=(seq_length, 1), kernel_regularizer=l2(lam), return_sequences=True),
    SimpleRNN(units=1, activation='linear', kernel_regularizer=l2(lam), return_sequences=True)
])


model.compile(optimizer=Adam(learning_rate=eta), loss='mse')
model.summary()

# Reshape
X = X[np.newaxis, :, np.newaxis]
y = y[np.newaxis, :, np.newaxis]

# Train the model
history = model.fit(X, y, epochs=epochs, batch_size=batch_size)

# Make predictions
y_pred = model.predict(X)

# Plot predictions vs true values
plt.figure()
plt.plot(y[0,:,0], "k", label='True Values')
plt.plot(y_pred[0,:], "r", label='Predictions')
plt.xlabel('Time Step')
plt.ylabel('Displacement')
plt.legend()
plt.savefig("Figures/HARM_OSC/tf_harm.pdf")

# Evaluate the model
loss = model.evaluate(X, y)
print(f'Test Loss: {loss}')




from RNN_module.RNN import RNN
from RNN_module.funcs import CostOLS, tanh, identity
from RNN_module.schedulers import Adam as my_Adam

scheduler = my_Adam(eta)
my_model = RNN(CostOLS, scheduler)
my_model.add_InputLayer(1)
my_model.add_RNNLayer(n_nodes_hidden, tanh)
my_model.add_OutputLayer(1, identity)
scores = my_model.train(X, y, epochs=epochs, batches=batches, lmbd=lam)
my_pred = my_model.feed_forward(X)


print("Tensorflow:")
print(f"    Training error = {model.evaluate(X, y)}")

print("My model:")
print(f"    Training error = {scores["train_error"][-1]}")

plt.figure()
plt.plot(y[0,:,0], "k", label='True Values')
plt.plot(my_pred[0,:,0], "r", label='Predictions')
plt.xlabel('Time Step')
plt.ylabel('Displacement')
plt.legend()
plt.savefig("Figures/HARM_OSC/my_harm.pdf")

plt.figure()
plt.plot(scores["train_error"], label="My error")
plt.plot(history.history["loss"], label="Tensorflow error")
plt.xlabel("Epochs")
plt.ylabel("Error")
plt.legend()
plt.savefig("Figures/HARM_OSC/error.pdf")