import numpy as np
import pandas as pd
from scipy.fft import fft, fft2, fftfreq, fftshift
import matplotlib.pyplot as plt

def wave(x,t):
    k = 1
    omega = np.sin(t-np.pi/2)
    return np.exp(1j*(k*x - omega*t))

class dim:
    def __init__(self, domain, step):
        self.domain = domain
        self.step = step
        self.points = np.arange(domain[0], domain[1], step)
        self.length = len(self.points)
        self.inverse = 2*np.pi*(fftfreq(self.length, step))*step
        self.range = domain[1] - domain[0]
    
def signal_plot(axis, xtick_count, ytick_count):
    plt.xlabel("$t / \pi$ s")
    plt.ylabel("$x$ / m")
    plt.title("Real part of wave")
    xlabels = [round(i/np.pi,2) for i in np.linspace(time.domain[0],time.domain[1],xtick_count)]
    plt.xticks(np.linspace(0,time.length,xtick_count).astype(int), xlabels) # (where to put ticks, what labels)
    plt.yticks(np.linspace(0,space.length,ytick_count).astype(int), np.linspace(space.domain[0],space.domain[1],ytick_count))

def ft_plot(xtick_count, ytick_count):    
    plt.xlabel("$\omega\ /\ \mathrm{rad\ s^{-1}}$")
    plt.ylabel("$k$ / m$^{-1}$")
    plt.title("Absolute value of 2D FFT")
    xlabels = [round(i,3) for i in time.inverse[np.round(np.linspace(0, time.length-1, xtick_count)).astype(int)]]
    ylabels = [round(i,3) for i in space.inverse[np.round(np.linspace(0, space.length-1, ytick_count)).astype(int)]]
    plt.xticks(np.linspace(0,time.length,xtick_count).astype(int), xlabels, rotation=90) # (where to put ticks, what labels)
    plt.yticks(np.linspace(0,space.length,ytick_count).astype(int), ylabels, rotation=0)

def get_closest(arr, val):
    dist = 1e6
    here = 0
    for i in arr:
        if abs(val-i) < dist:
            dist = abs(val-i)
            idx = here
        here += 1
    return idx

def ft_zoom_plot(omega_lims, k_lims):
    omega_idx_lims = (get_closest(time.inverse, omega_lims[0]), get_closest(time.inverse, omega_lims[1]))
    k_idx_lims = (get_closest(space.inverse, k_lims[0]), get_closest(space.inverse, k_lims[1]))
    plt.imshow(np.abs(transform.iloc[k_idx_lims[0]:k_idx_lims[1],omega_idx_lims[0]:omega_idx_lims[1]]), aspect=100)
    plt.xlabel("$\omega\ /\ \mathrm{rad\ s^{-1}}$")
    plt.ylabel("$k$ / m$^{-1}$")
    plt.title("Absolute value of 2D FFT")
    xlabels = [round(i,3) for i in time.inverse[omega_idx_lims[0]:omega_idx_lims[1]]]
    ylabels = [round(i,3) for i in space.inverse[k_idx_lims[0]:k_idx_lims[1]]]
    plt.xticks([i for i in range(omega_idx_lims[1]-omega_idx_lims[0])], xlabels, rotation=90)
    plt.yticks([i for i in range(k_idx_lims[1]-k_idx_lims[0])], ylabels, rotation=0)

def plot_line(k_lims, aspect, xtick_count):
    omega_idx_lims = (0, len(time.inverse)-1)
    k_idx_lims = (get_closest(space.inverse, k_lims[0]), get_closest(space.inverse, k_lims[1]))
    plt.imshow(np.abs(transform.iloc[k_idx_lims[0]:k_idx_lims[1],omega_idx_lims[0]:omega_idx_lims[1]]), aspect=aspect)
    plt.xlabel("$\omega\ /\ \mathrm{rad\ s^{-1}}$")
    plt.ylabel("$k$ / m$^{-1}$")
    plt.title("Absolute value of 2D FFT")
    xlabels = [round(i,3) for i in time.inverse[np.round(np.linspace(0, time.length-1, xtick_count)).astype(int)]]
    ylabels = [round(i,3) for i in space.inverse[k_idx_lims[0]:k_idx_lims[1]]]
    plt.xticks(np.linspace(0,time.length,xtick_count).astype(int), xlabels, rotation=90) # (where to put ticks, what labels)
    plt.yticks([i for i in range(k_idx_lims[1]-k_idx_lims[0])], ylabels, rotation=0)

def get_k_line(k):
    index = get_closest(space.inverse, k)
    line = np.abs(transform[:][index:index+1])
    figs = [line[i][index] for i in line]
    plt.plot(fftfreq(time.length,(time.domain[1]-time.domain[0])/time.length),np.abs(fft(figs)))
    plt.xlabel("Frequency")
    plt.ylabel("$|F(1,\omega)|$")
    #plt.xlim([-3,3])
    plt.savefig("omega_line1.pdf")

space = dim((0,99), 0.1)
time = dim((0,32*np.pi), 0.01)

# Load value of wave at points into 2d array.
mat = []
for xi in space.points:
    tmp = []
    for ti in time.points:
        tmp.append(wave(xi,ti))
    mat.append(tmp)

# Create dataframe with dimensions (space.length, time.length)
signal = pd.DataFrame(mat, dtype = complex)

ax = plt.imshow(np.real(signal),aspect=5)
signal_plot(ax, 10, 5)
plt.show()

transform = pd.DataFrame(fft2(signal), dtype = complex)
ax2 = plt.imshow(np.abs(transform))
ft_plot(10,10)
plt.show()

ft_zoom_plot((0.995,1.005), (0.95,1.05))
plt.show()

plot_line((0.95,1.05), 20, 20)
plt.show()

#get_k_line(1)