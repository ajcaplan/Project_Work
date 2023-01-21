#!/usr/bin/env python
# coding: utf-8

# # Fourier analysis of BES data
#
# Aim: obtain a graph of angular frequency against wavenumber.
# Need to run 'export HDF5_USE_FILE_LOCKING=FALSE' in terminal first to avoid HDF error

# Import neccesary packages

# In[1]:

print("Start")
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from plotting_functions_BES import *
print("Packages installed")

# Read BES data

# In[2]:


# for BES data
# (R, z) locations for the BES channels (view location)
apdpos = np.asarray(xr.open_dataarray('/shared/storage/plasma/turb_exp/ajc649/Data/shot29378_apdpos.nc'))
fluct_data_from_file = xr.open_dataarray('/shared/storage/plasma/turb_exp/ajc649/Data/shot29378_LH_fluct_data.nc')

bes_time = np.asarray(fluct_data_from_file.coords['time']) # bes_time[0:992499]
fluct_data = np.asarray(fluct_data_from_file) # fluct_data[0:31][0:992499]
print("Data gathered")

# Plot some fluctiations for illustration.

# In[18]:


#plot_bes_fluctuations(29378, bes_time, fluct_data, [bes_time[0], bes_time[-1]], [0,1,2,3], "fluct_plot")
print("Example fluctuations plotted")

# FFT in time of one example channel to illustrate.

# In[6]:


"""channel = 0
ffts = get_channel_fft(29378, bes_time, fluct_data, channel, [bes_time[0], bes_time[-1]], "channel_fft")
plt.plot(ffts[0], ffts[1],color='k')
plt.xlabel('frequency [kHz]', fontsize=22)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.xscale('log')
plt.xlim([10, 1.0e3])
plt.yscale('log')
plt.title(('shot29378 ch ' + str(channel) + ', times ' + str([round(bes_time[0], 7), round(bes_time[-1], 7)])), fontsize=32)
#plt.show()
plt.savefig(('./Plots/shot29378_ch' + str(channel) + '_spectrum_example.png'), format='png', transparent=False, bbox_inches="tight")
plt.close()"""


# Generate array containing each channel's FFT over frequency space and an array of frequencies.
#
# Then get arrays of focal points of channels.

# In[19]:

regions = [[0.16,0.24], [0.26,0.54], [0.54,0.68]]

#figure, axes = plt.subplots(3,1,sharex=True, figsize=(15,9))
#axes[0].set_title("k-f plots")

for region in range(3):
    print("Region", region+1)
    print("FFTing each channel")
    channel_freqs = []
    spec = []

    for i in range(0,25,8):
        fft = get_channel_fft(29378, bes_time, fluct_data, i, regions[region], "channel_fft")
        freqs, abs_square = fft[0], fft[1]
        spec.append(abs_square) # Each row of spec corresponds to a channel.
    channel_freqs = freqs

    spec = np.asarray(spec)
    spec = np.transpose(spec) # Now each row is a time point as required by calc_kspecs

    # Get coords of left-most column
    R_positions = np.asarray([i[0] for i in apdpos[::8]])
    Z_positions = np.asarray([i[0] for i in apdpos[::8]])


    # Feed into <code>calc_kspec</code>

    # In[20]:

    print("Calculating kfspec")
    calc = calc_kspec(spec, R_positions)
    kf_matrix = calc[0]
    k_arr = calc[1]
    print(R_positions.shape)
    print(kf_matrix.shape)
    print(k_arr.shape)

    # In[18]:

    print("Plotting k-f graph")
    klabels = [round(i,3) for i in k_arr[np.round(np.linspace(0, len(k_arr)-1, 14)).astype(int)]]
    flabels = [round(i,3) for i in channel_freqs[np.round(np.linspace(0, len(channel_freqs)-1, 14)).astype(int)]]
    
    plt.matshow(np.log(np.abs(kf_matrix)), aspect=len(k_arr)/len(freqs))
    #plt.title("Region " + str(region+1))
    plt.ylabel("Frequency")
    plt.yticks(np.linspace(0,len(kf_matrix),14).astype(int), flabels)
    plt.xticks(np.linspace(0,len(kf_matrix[0]),14).astype(int), klabels, rotation=90)
    plt.xlabel("Wavenumber")

    plt.savefig("./Plots/fk_plot_small_reg_"+str(region+1)+".pdf", format="pdf", bbox_inches="tight")
    plt.close()

# In[19]:
print("End")
exit()
