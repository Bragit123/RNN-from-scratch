import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from funcs import identity, CostOLS
from sklearn.preprocessing import MinMaxScaler
from RNN import RNN
from schedulers import Adam

cost_func = CostOLS
act_func_hidden = identity
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
x_orig = weather_data["tmax"][:N].to_numpy()
t_orig = weather_data["tmax_tomorrow"][:N].to_numpy()

x = x_orig[:, np.newaxis]
t = t_orig[:, np.newaxis]
sc = MinMaxScaler()
sc = sc.fit(x)

x = sc.transform(x)
t = sc.transform(t)

# Give data correct dimensions
x = x[np.newaxis,:]
t = t[np.newaxis,:]

## Train network on data
epochs = 50
batches = 1
scores = rnn.train(x, t, epochs=epochs, batches=batches, lmbd = 0.001, store_output=True)

## Extract output
epoch_arr = np.arange(1, epochs+1)
y_history_scaled = scores["y_train_history"]
train_error = scores["train_error"]

## Inversely scale output
y_history = np.zeros(y_history_scaled.shape)
for i in range(y_history_scaled.shape[0]):
    yi = y_history_scaled[i,0,:,:]
    yi_scaleback = sc.inverse_transform(yi)
    y_history[i,0,:,:] = yi_scaleback

## Plot error during training
plt.figure()
plt.plot(epoch_arr, train_error)
plt.yscale("log")
plt.savefig("weather_error.pdf")

## Plot output and target after training
seq_ind = np.arange(N)
y = y_history[-1,0,:,0]
plt.figure()
plt.plot(seq_ind, t_orig, "k", label="Target")
plt.plot(seq_ind, y, "b--", label="Output")
plt.legend()
plt.savefig("weather.pdf")

## Create animation of how the output fits to the target
y = y_history[0,0,:,0]
fig, ax = plt.subplots()
t_plot = ax.plot(seq_ind, t_orig, "k", label="Target")[0]
y_plot = ax.plot(seq_ind, y, "b--", label="Output")[0]
epoch_text = ax.text(5, 70, f"Epoch: {0}")
cost_text = ax.text(5, 68, f"Cost: {train_error[0]:.2f}")
ax.set(xlim=[0,N], ylim=[np.min(t_orig)-1,np.max(t_orig)+1])
ax.legend()

def update_plot(frame):
    y = y_history[frame,0,:,0]
    y_plot.set_ydata(y)
    epoch_text.set_text(f"Epoch: {frame}")
    cost_text.set_text(f"Cost: {train_error[frame]:.2f}")
    return y_plot

anim = animation.FuncAnimation(fig=fig, func=update_plot, frames=epochs, interval=100)
anim.save("weather_fit.gif", writer="pillow")