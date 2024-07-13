import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim

n = 100
N = 1000
a = np.linspace(0, 4, n)
x = np.linspace(0, 20, N)

fig, ax = plt.subplots()
line = ax.plot(x, a[0]*x)[0]
ax.set(xlim=[0,20], ylim=[-1,20])

def update(frame):
    y = a[frame] * x
    line.set_ydata(y)
    return line

ani = anim.FuncAnimation(fig=fig, func=update, frames=n, interval=10)
ani.save("mymovie.mp4", writer="pillow")