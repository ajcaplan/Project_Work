from functions_BES import *

x = np.linspace(0,16*np.pi,1000)
y = np.sin(x) + np.cos(2*x)

fig, ax1 = plt.subplots()
ax1.plot(x,y)
plt.show()

fn = calc_kspec(y, x)
ax2 = plt.plot(fn[1], fn[0])
plt.xlim(left=-5,right=5)
plt.show()