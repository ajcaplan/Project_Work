import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from scipy.fft import fft

# for density, Dalpha and upper tangential Dalpha
dalpha_from_file = xr.open_dataarray('shot29378_dalpha.nc')

dalpha_time = np.asarray(dalpha_from_file.coords['time'])
dalpha_data = np.asarray(dalpha_from_file)

# for equilibria
equilib_from_file = xr.open_dataarray('shot29378_equilibria.nc')

equilib_time = np.asarray(equilib_from_file.coords['time'])
equilib_R = np.asarray(equilib_from_file.coords['R'])
equilib_Z = np.asarray(equilib_from_file.coords['Z'])
equilib_psi = np.asarray(equilib_from_file)

"""
# could then plot flux surfaces for a particular timepoint (i.e. eq_idx)
index = 0
for i in range(len(equilib_time)):
    plt.contour(equilib_R, equilib_Z, equilib_psi[index], np.linspace(0, 1, 11))
    plt.savefig("frames/Frame" + str(index) + ".png")
    plt.close()
    index += 1
"""

# for BES data
# (R, z) locations for the BES channels (view location)
apdpos = np.asarray(xr.open_dataarray('shot29378_apdpos.nc'))

fluct_data_from_file = xr.open_dataarray('shot29378_LH_fluct_data.nc')

bes_time = np.asarray(fluct_data_from_file.coords['time'])
fluct_data = np.asarray(fluct_data_from_file)
# fluct_data has 4 rows, 8 channels (mapped to apdpos)

for i in range(32):
    #plt.plot(fft(fluct_data[i]))
    plt.plot(bes_time, fluct_data[i])
    plt.title("Channel " + str(i) + ", " + str(apdpos[i]))
    plt.show()

"""
# Generate and plot 8x4 array of flucation data at a given time
time_idx=0
for i in range(len(bes_time)):
    tmp = []
    for t in fluct_data:
        tmp.append(t[time_idx])
    tmp = np.reshape(tmp, (4,8))
    plt.matshow(tmp)
    plt.savefig("frames/Frame" + str(time_idx) + ".png")
    plt.close()
    time_idx+=1
"""