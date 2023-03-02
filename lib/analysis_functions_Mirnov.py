from lib.imports_BES import *
from lib.analysis_functions_BES import *

def get_kf_spec_mirnov(mirnov_time, mirnov_data, coilpos, timeslice):    
    space_array = [coilpos[i][1] for i in range(len(coilpos))] # Get Z-coordinates of each coil
    space_array = np.asarray(space_array)
    spec = []
    for coil in range(len(coilpos)): # FFT each coil in time
        fft = fft_channel(mirnov_time, mirnov_data, coil, timeslice)
        f_transform = fft[1]
        spec.append(f_transform) # Each row of spec corresponds to a coil.
    f_arr = fft[0] # Frequency array is the same for all coil so just save any one.
    
    spec = np.asarray(spec)
    spec = np.transpose(spec) # Now each row is a time point as required by calc_kspec
    calc = calc_kspec(spec, space_array) # Get k-f spectrum
    
    kf_matrix = calc[0] # This contains the transform data
    k_arr = calc[1] # This is the array of wavenumbers
    
    return f_arr, k_arr, kf_matrix


# Adds STFTs for all coils and optionally plots result with vlines for showing windows
def sum_mirnov_fluct_spectrogram(shot, mirnov_time, mirnov_data, timeslice, n=8, freq_lims=[0.0, 200.0], vlines=None, plot=True, save=False):
    # plot the BES fluctuation data spectrogram for one or more channels, L. Howlett adapted by A. Caplan
    idx1 = (np.abs(mirnov_time - timeslice[0])).argmin()
    idx2 = (np.abs(mirnov_time - timeslice[1])).argmin()
    
    freq, times, Sxx = sig.spectrogram(mirnov_data[:,idx1:idx2], fs=1/np.mean(np.diff(mirnov_time[idx1:idx2])), 
                                       nperseg=(2 ** n), scaling='spectrum')
    
    new_lim = len(freq)#int(14 * (2 ** (n - 7)))
    summed_Sxx = np.sum(np.asarray(Sxx[:][:new_lim,:]), axis=0)
    if plot == True:
        figure, axes = plt.subplots(1, 1, sharex=True, 
                                    figsize=(15, 4))
        axes.set_title("\#" + str(shot) + " spectrogram, n=" + 
            str(int(2 ** n)))

        levs = [10**i for i in range(-11,2)]
        ct = axes.contourf(times + mirnov_time[idx1], 0.001 * freq[:new_lim], 
               summed_Sxx[:new_lim,:], 16, cmap=plt.get_cmap('plasma'), levels=levs,
                norm=(colors.LogNorm()))

        if vlines != None:
            for line in vlines:
                axes.vlines(line[0], 0, 200, "green", linestyle="dashed", linewidth=0.8)
                axes.vlines(line[1], 0, 200, "red", linestyle="dashed", linewidth=0.8)
        axes.set_ylim(freq_lims)
        axes.set_ylabel(r"$f$ [kHz]")
        cbar = figure.colorbar(ct, shrink=0.9, label="Strength [a.u.]")
        axes.set_xlabel("Time [s]")
        axes.set_xlim(timeslice)
        plt.tight_layout()

        if save != False:
            plt.savefig(save + ".png", format="png", bbox_inches="tight", dpi=300)
        else:
            plt.show()
        plt.close()
    return freq, times + mirnov_time[idx1], summed_Sxx