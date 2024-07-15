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
val_start = 1000
N_val = 70
weather_data = pd.read_csv("../Data/clean_weather.csv")
x_orig = weather_data["tmax"][:N].to_numpy()
t_orig = weather_data["tmax_tomorrow"][:N].to_numpy()
x_val_orig = weather_data["tmax"][val_start:val_start+N_val].to_numpy()
t_val_orig = weather_data["tmax_tomorrow"][val_start:val_start+N_val].to_numpy()

x = x_orig[:, np.newaxis]
t = t_orig[:, np.newaxis]
x_val = x_val_orig[:,np.newaxis]
t_val = t_val_orig[:,np.newaxis]
sc = MinMaxScaler()
sc = sc.fit(x)

x = sc.transform(x)
t = sc.transform(t)
x_val = sc.transform(x_val)
t_val = sc.transform(t_val)

# Give data correct dimensions
x = x[np.newaxis,:]
t = t[np.newaxis,:]
x_val = x_val[np.newaxis,:]
t_val = t_val[np.newaxis,:]

## Train network on data
epochs = 50
batches = 1
scores = rnn.train(x, t, x_val, t_val, epochs=epochs, batches=batches, lmbd = 0.001, store_output=True)

## Extract output
epoch_arr = np.arange(1, epochs+1)
y_history_scaled = scores["y_train_history"]
train_error = scores["train_error"]
y_val_history_scaled = scores["y_val_history"]
val_error = scores["val_error"]

## Inversely scale output
y_history = np.zeros(y_history_scaled.shape)
for i in range(y_history_scaled.shape[0]):
    yi = y_history_scaled[i,0,:,:]
    yi_scaleback = sc.inverse_transform(yi)
    y_history[i,0,:,:] = yi_scaleback

## Plot error during training
plt.figure()
plt.plot(epoch_arr, train_error, label="Training error")
plt.plot(epoch_arr, val_error, label="Validation error")
plt.yscale("log")
plt.legend()
plt.savefig("weather_error.pdf")

## Plot output and target after training
seq_ind = np.arange(N)
y = y_history[-1,0,:,0]
plt.title("Maximum temperature over time.")
plt.xlabel("Number of days")
plt.ylabel("Temperature")
plt.figure()
plt.plot(seq_ind, t_orig, "k", label="Target")
plt.plot(seq_ind, y, "b--", label="Output")
plt.text(5, 70, f"Error: {train_error[-1]:.2f}")
plt.legend()
plt.savefig("weather.pdf")

## Extrapolate data using network
length = 50
y_extra = rnn.extrapolate(length)
y_extra = y_extra[0,:,:]
y_extra = sc.inverse_transform(y_extra)
y_extra = y_extra[:,0]
seq_extra = np.arange(N, N+length)
t_extra = weather_data["tmax_tomorrow"][N:N+length].to_numpy()
plt.figure()
plt.title("Maximum temperature over time.")
plt.xlabel("Number of days")
plt.ylabel("Temperature")
plt.plot(seq_ind, t_orig, "k", label="Target")
plt.plot(seq_extra, t_extra, "k:")
plt.plot([seq_ind[-1],seq_extra[0]], [t_orig[-1],t_extra[0]], "k--")
plt.plot(seq_ind, y, "b--", label="Output")
plt.plot([seq_ind[-1],seq_extra[0]], [y[-1],y_extra[0]], "r--")
plt.plot(seq_extra, y_extra, "r--", label="Extrapolation")
plt.yscale("log")
plt.legend()
plt.savefig("weather_extra.pdf")

## Plot validation results
seq_val = np.arange(N_val)
y_val = y_val_history_scaled[-1,0,:,:]
y_val = sc.inverse_transform(y_val)
y_val = y_val[:,0]
plt.figure()
plt.title("Maximum temperature over time.")
plt.xlabel("Number of days")
plt.ylabel("Temperature")
plt.plot(seq_val, t_val_orig, "k", label="Target")
plt.plot(seq_val, y_val, "b--", label="Output")
plt.text(1005, 55, f"Error: {val_error[-1]:.2f}")
plt.legend()
plt.savefig("weather_validation.pdf")


## Create animation of how the output fits to the target
y = y_history[0,0,:,0]
fig, ax = plt.subplots()
ax.set_title("Maximum temperature over time.")
ax.set_xlabel("Number of days")
ax.set_ylabel("Temperature")
t_plot = ax.plot(seq_ind, t_orig, "k", label="Target")[0]
y_plot = ax.plot(seq_ind, y, "b--", label="Output")[0]
epoch_text = ax.text(5, 70, f"Epoch: {0}")
error_text = ax.text(5, 68, f"Error: {train_error[0]:.2f}")
ax.set(xlim=[0,N], ylim=[np.min(t_orig)-1,np.max(t_orig)+1])
ax.legend()

def update_plot(frame):
    y = y_history[frame,0,:,0]
    y_plot.set_ydata(y)
    epoch_text.set_text(f"Epoch: {frame}")
    error_text.set_text(f"Error: {train_error[frame]:.2f}")
    return y_plot

anim = animation.FuncAnimation(fig=fig, func=update_plot, frames=epochs, interval=100)
anim.save("weather_anim.gif", writer="pillow")


## Create animation for validation results
## Inversely scale output
y_val_history = np.zeros(y_val_history_scaled.shape)
for i in range(y_val_history_scaled.shape[0]):
    yi = y_val_history_scaled[i,0,:,:]
    yi_scaleback = sc.inverse_transform(yi)
    y_val_history[i,0,:,:] = yi_scaleback

y_val = y_val_history[0,0,:,0]
fig, ax = plt.subplots()
ax.set_title("Maximum temperature over time.")
ax.set_xlabel("Number of days")
ax.set_ylabel("Temperature")
t_plot = ax.plot(seq_val, t_val_orig, "k", label="Target")[0]
y_plot = ax.plot(seq_val, y_val, "b--", label="Output")[0]
epoch_text = ax.text(5, 55, f"Epoch: {0}")
error_text = ax.text(5, 53, f"Error: {train_error[0]:.2f}")
ax.set(xlim=[0,N_val], ylim=[np.min(t_val_orig)-1,np.max(t_val_orig)+1])
ax.legend()

def update_plot(frame):
    y_val = y_val_history[frame,0,:,0]
    y_plot.set_ydata(y_val)
    epoch_text.set_text(f"Epoch: {frame}")
    error_text.set_text(f"Error: {train_error[frame]:.2f}")
    return y_plot

anim = animation.FuncAnimation(fig=fig, func=update_plot, frames=epochs, interval=100)
anim.save("weather_val_anim.gif", writer="pillow")