import numpy as np
import seaborn
import pandas as pd
from scipy.fft import fft2, fftfreq
import matplotlib.pyplot as plt

def wave(x,t):
    k = 1
    omega = np.sin(t)
    return np.exp(1j*(k*x - omega*t))

tpoints = 1000
t_end = 32*np.pi
t = np.linspace(0, t_end, tpoints)

xpoints = 200
x = np.linspace(0,99, xpoints)

mat = []
for xi in x:
    tmp = []
    for ti in t:
        tmp.append(wave(xi,ti))
    mat.append(tmp)
    
signal = pd.DataFrame(mat, dtype = complex)
s1 = seaborn.heatmap(np.real(signal))
plt.xlabel("$t / \pi$ s")
plt.ylabel("$x$ / m")
plt.title("Real part of wave")

s1.set_xticks(range(len(t))[::10], np.round(t[::10]/np.pi,3))
s1.set_yticks(range(len(x))[::10], np.round(x[::10],3))


plt.savefig("Signal.pdf",bbox_inches='tight')
plt.show()

transformed  = pd.DataFrame(fft2(signal), dtype = complex)
ax = seaborn.heatmap(np.abs(transformed))
plt.xlabel("$\omega\ /\ \mathrm{rad\ s^{-1}}$")
plt.ylabel("$k$ / m$^{-1}$")
plt.title("Absolute value of 2D FFT")

ax.set_xticks(range(len(t))[::int(round(0.1*tpoints,2))])
ax.set_xticklabels([np.round(2*np.pi*i,3) for i in fftfreq(tpoints,t_end/tpoints)[::int(round(0.1*tpoints,2))]])
ax.set_yticks(range(len(x))[::10])
ax.set_yticklabels([np.round(2*np.pi*i,3) for i in fftfreq(xpoints,100/xpoints)[::10]])

plt.savefig("fft2.pdf",bbox_inches='tight')
plt.show()

"""zoom = seaborn.heatmap(np.abs(transformed.iloc[10:20,190:])) # .iloc[0:10,80:90]
plt.xlabel("$\omega\ /\ \mathrm{rad\ s^{-1}}$")
plt.ylabel("$k$ / m$^{-1}$")
plt.title("Absolute value of 2D FFT")

zoom.set_xticks(range(len(t[190:])))
zoom.set_xticklabels([np.round(2*np.pi*i,3) for i in fftfreq(tpoints,t_end/tpoints)[190:]], rotation=90)
zoom.set_yticks(range(len(x[10:20])))
zoom.set_yticklabels([np.round(2*np.pi*i,3) for i in fftfreq(xpoints,100/xpoints)[10:20]], rotation = 0)

plt.savefig("zoomed_fft2.pdf",bbox_inches='tight')
plt.show()"""