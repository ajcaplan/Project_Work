from lib.imports_BES import *

# Work out average distance of a BES column from the separatrix at a given time or over a given timeslice
def sol_dist(equilib_time, equilib_R, equilib_Z, equilib_psi, apdpos, timeslice, col):   
    # Only looking for points in the relevant z and right of array's left-most point
    # Add on bits to ensure the contour extends far enough
    zmin = np.abs(np.min(apdpos[:,1])-equilib_Z-0.1).argmin()
    zmax = np.abs(np.max(apdpos[:,1])-equilib_Z+0.1).argmin()
    rmin = np.abs(np.min(apdpos[:,0])-equilib_R-0.1).argmin()
    
    # Convert timeslice to indices of equilib_time
    if isinstance(timeslice, list) or isinstance(timeslice, np.ndarray):
        idx1 = np.abs(equilib_time-timeslice[0]).argmin()
        idx2 = np.abs(equilib_time-timeslice[1]).argmin()
    else:
        idx1 = np.abs(equilib_time-timeslice).argmin()
        idx2 = idx1
    
    # Get coordinates of each channel in the specified column
    ch_pos = [apdpos[ch] for ch in np.arange(col,32,8)]
    
    all_time = [] # Stores averages distances for each time
    for time in range(idx1, idx2+1):
        # Get path of separatrix
        equilib_psi_t = equilib_psi[time] # Data at this time
        CS = plt.contour(equilib_R[rmin:], equilib_Z[zmin:zmax], equilib_psi_t[zmin:zmax,rmin:], [1.0])
        plt.close()
        path = mplPath.Path(CS.allsegs[0][0])

        # Extrapolate to get more coordinates
        verts = path.vertices
        verts_expanded = []
        idx = 0
        for i in verts[:-1]:
            for r in np.linspace(verts[idx,0],verts[idx+1,0],100): # Get lots of points
                verts_expanded.append([r, np.interp(r, verts[idx:idx+2,0], verts[idx:idx+2,1])])
            idx += 1
        verts_expanded = np.asarray(verts_expanded)

        # Find r-coord of point closest in z to channel and find dist to separatrix
        this_time = []
        for ch in ch_pos:
            closest_r = verts_expanded[np.abs(verts_expanded[:,1]-ch[1]).argmin(),0]
            this_time.append(ch[0] - closest_r)
        all_time.append(np.mean(this_time))
    return np.mean(all_time)

# Use upper tangential Dalpha to work out times of crashes in a given timeslice
# based on a threshold maximum worked out in the main Notebook
def get_crash_times(utda_time, utda_data, threshold, timeslice):
    timestep = np.mean(np.diff(utda_time))
    idx_jump = int(0.0015//timestep)
    idx1 = (np.abs(utda_time - timeslice[0])).argmin()
    idx2 = (np.abs(utda_time - timeslice[1])).argmin()
    ddata = np.gradient(utda_data)
    
    check_idx = idx1
    windows = []
    while check_idx <= idx2:
        if ddata[check_idx] > threshold:
            windows.append([check_idx-idx_jump, check_idx+idx_jump])
            check_idx += idx_jump
        else:
            check_idx += 1
    
    crash_times = []
    for crash in windows:
        max_idx = crash[0] + np.where(ddata[crash[0]:crash[1]] == np.max(ddata[crash[0]:crash[1]]))[0][0]
        crash_times.append(utda_time[max_idx])
    return crash_times

# Define timeslices next to the crash based on given time shifts
def get_crash_adjacent_window(utda_time, utda_data, threshold, timeslice, start_shift=5e-3, end_shift=1e-3):
    crash_times = get_crash_times(utda_time, utda_data, threshold, timeslice)
    windows = []
    
    for i in crash_times:
        windows.append([i-start_shift, i-end_shift])
        
    return windows

# Add BES spectrograms within a column. Should reduce background in the plots.
def sum_bes_fluct_spectrogram(shot, bes_time, fluct_data, col, timeslice, n=8, freq_lims=[0.0, 200.0], vlines=None, plot=True, save=False):
    # plot the BES fluctuation data spectrogram for one or more channels, L. Howlett adapted by A. Caplan
    idx1 = (np.abs(bes_time - timeslice[0])).argmin()
    idx2 = (np.abs(bes_time - timeslice[1])).argmin()
    channels = []
    for i in range(4): # For the given column, get each channel index
        channels.append(i*8+col)
    
    freq, times, Sxx = sig.spectrogram(fluct_data[:,idx1:idx2], fs=f_samp, 
                                       nperseg=(2 ** n), scaling='spectrum')
    
    new_lim = int(14 * (2 ** (n - 7)))
    summed_Sxx = np.asarray(Sxx[channels[0]][:new_lim,:])
    for ch in channels[1:]:
        summed_Sxx += Sxx[ch][:new_lim,:]
    
    if plot == True:
        figure, axes = plt.subplots(1, 1, sharex=True, 
                                    figsize=(15, 4))
        axes.set_title("\#" + str(shot) + " spectrogram, sum of col " + str(col+1) + ", n=" + 
            str(int(2 ** n)))    

        ct = axes.contourf(times + bes_time[idx1], 0.001 * freq[:new_lim], 
               summed_Sxx, 16, cmap=plt.get_cmap('plasma'), levels=[10**(i/10) for i in range(-80,10,5)],
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
    return freq, times + bes_time[idx1], summed_Sxx

# Calculate the FFT of one channel of BES data in time
def fft_channel(bes_time, fluct_data, ch, timeslice):
    idx1 = (np.abs(bes_time - timeslice[0])).argmin()
    idx2 = (np.abs(bes_time - timeslice[1])).argmin()
    
    ps_bes = np.fft.fft(fluct_data[ch][idx1:idx2])
    time_step_bes = np.mean(np.diff(bes_time[idx1:idx2]))
    freqs_bes = np.fft.fftfreq(fluct_data[ch][idx1:idx2].size, time_step_bes)
    idx_bes = np.argsort(freqs_bes)
    
    return freqs_bes[idx_bes], ps_bes[idx_bes]

# Calculate an FFT in space
def calc_kspec(spec, xvec, odd=1, decimals=1, antialias=1., direction=1.):
    # from I. Cziegler
    xsort = np.sort(xvec)
    dxsort = np.sort(np.diff(xsort))
    dx = dxsort[0]
    n = 0
    while dx == 0:
        n = n + 1
        dx = dxsort[n]
    kmax = np.pi / dx * antialias
    kmin = np.floor((10 ** decimals) * np.pi / 
                    (xvec.max() - xvec.min()) / 2) / (10 ** decimals)
    while kmin == 0:
        decimals=decimals + 1
        kmin = np.floor((10 ** decimals) * np.pi / 
                        (xvec.max() - xvec.min()) / 2) / (10 ** decimals)
    klen = int(2 * np.around(kmax / kmin) + odd)
    k_arr = np.linspace(-kmax, kmax, klen)
    fcomps = np.exp(1j * direction * np.outer(xvec, k_arr))
    kfspec = np.dot(spec, fcomps)
    return kfspec, k_arr

# Use calc_kspec and fft_channel to 2D FFT BES data over a given timeslice
def get_kf_spec(bes_time, fluct_data, apdpos, col, timeslice): # Col is int from 0 to 7 . timeslice = [t1, t2], a 2x1 array
    channels = []
    for i in range(4): # For the given column, get each channel index
        channels.append(i*8+col)
    
    space_array = [apdpos[i][1] for i in channels] # Get Z-coordinates of each channel
    space_array = np.asarray(space_array)
    spec = []
    for ch in channels: # FFT each channel in time
        fft = fft_channel(bes_time, fluct_data, ch, timeslice)
        f_transform = fft[1]
        spec.append(f_transform) # Each row of spec corresponds to a channel.
    f_arr = fft[0] # Frequency array is the same for all channels so just save any one.
    
    spec = np.asarray(spec)
    spec = np.transpose(spec) # Now each row is a time point as required by calc_kspec
    calc = calc_kspec(spec, space_array) # Get k-f spectrum
    
    kf_matrix = calc[0] # This contains the transform data
    k_arr = calc[1] # This is the array of wavenumbers
    
    return f_arr, k_arr, kf_matrix

# Sum together |S(f,k)| for all crash-adjacent windows in a given timeslice
def kf_spec_sum_windows(bes_time, fluct_data, apdpos, col, timeslices):
    first_kf = get_kf_spec(bes_time, fluct_data, apdpos, col, timeslices[0])
    kf_summed = np.abs(first_kf[2])
    for window in range(1,len(timeslices)):
        kf_summed += np.abs(get_kf_spec(bes_time, fluct_data, apdpos, col, timeslices[window])[2])
    return first_kf[0], first_kf[1], kf_summed


# DEPRECATED: plot_kf_spec. Use instead plot_dispersion_relation(...) in analysis_functions_general
# Main function for plotting dispersion relations. Arguments offer control of plot type and saving
def plot_kf_spec(f_arr, k_arr, kf_matrix, plot_title, fint=50.0, fmin=0.0, fmax=None, smooth_pts=None, conditional=False, save=False):
    if fmax == None:
        fmax = np.max(f_arr)
        
    if smooth_pts != None:
        line_idx = np.abs(f_arr - fmin).argmin()
        this_block = np.sum(np.abs(kf_matrix[line_idx:line_idx+smooth_pts]), axis=0)
        smooth_spec = np.array(this_block)
        smooth_freqs = np.array(f_arr[line_idx])
        line_idx += 1
        
        while line_idx < np.abs(f_arr - fmax).argmin() + smooth_pts:
            this_block = np.sum(np.abs(kf_matrix[line_idx:line_idx+smooth_pts]), axis=0)
            smooth_spec = np.vstack((smooth_spec, this_block))
            smooth_freqs = np.append(smooth_freqs, f_arr[line_idx])
            line_idx += 1
        kf_matrix = smooth_spec
        f_arr = smooth_freqs
    if conditional == True:
        kf_matrix = np.transpose(np.transpose(kf_matrix)/np.sum(kf_matrix,axis=1))
        cbar_label = r"$\log\vert S(k|f)\vert^2$"
    else:
        cbar_label = r"$\log\vert S(f,k)\vert^2$"
    
    # Only need to plot a section of the spectrum. At least half not needed.
    kf_matrix = kf_matrix[(np.abs(f_arr - fmin)).argmin():(np.abs(f_arr - fmax)).argmin()]
    f_arr = f_arr[(np.abs(f_arr - fmin)).argmin():(np.abs(f_arr - fmax)).argmin()]

    # Convert to array with rounded frequencies for easier of plotting.
    kf_matrix_plot = pd.DataFrame(kf_matrix, index=np.around(f_arr*1e-3,0), columns=np.around(k_arr,1))

    # For only plotting tick at every 50 kHz
    interval = int(np.abs(f_arr - fint).argmin())

    # Plot log of values so small enough range for features to be visible
    sns.heatmap(np.log(np.abs(kf_matrix_plot)**2)[::-1], yticklabels=interval, cmap="plasma", cbar_kws={"label": cbar_label})
    plt.title(plot_title)
    plt.ylabel("Frequency [kHz]")
    plt.xlabel("Wavenumber [m$^{-1}$]")
    if save != False:
        plt.savefig(save + ".png", format="png", bbox_inches="tight", dpi=300)
    else:
        plt.show()
    plt.close()


# Investigate cross correlation of particular frequencies with time
def cross_corr_fbands(shot, bes_time, fluct_data, col, timeslice, band1, band2,
                      n=8, freq_lims=[0.0,200.0], plot_spec=False, plot_cc=True, save_spec=False, save_cc=False):
    freqs, times, Sxx = sum_bes_fluct_spectrogram(shot, bes_time, fluct_data, col, timeslice, n=n, plot=False)
    
    band1_idx = (np.abs(freqs-band1)).argmin()
    band2_idx = (np.abs(freqs-band2)).argmin()
    
    band1_freq = freqs[band1_idx]
    band2_freq = freqs[band2_idx]
    
    band1_data = Sxx[band1_idx,:]
    band2_data = Sxx[band2_idx,:]

    if plot_spec == True:
        new_lim = int(14 * (2 ** (n - 7)))
        figure, axes = plt.subplots(1, 1, sharex=True, 
                                    figsize=(15, 4))
        axes.set_title("\#" + str(shot) + " spectrogram, sum of col " + str(col+1) + ", n=" + 
            str(int(2 ** n)))    
        ct = axes.contourf(times, 0.001 * freqs[:new_lim], 
            Sxx, 16, cmap=plt.get_cmap('plasma'), levels=[10**(i/10) for i in range(-80,10,5)],
            norm=(colors.LogNorm()))
        axes.hlines(freqs[band1_idx]*1e-3, times[0], times[-1], "cyan", linestyle="dashed", linewidth=1)
        axes.hlines(freqs[band2_idx]*1e-3, times[0], times[-1], "cyan", linestyle="dashed", linewidth=1)
        axes.set_ylim(freq_lims)
        axes.set_ylabel(r"$f$ [kHz]")
        cbar = figure.colorbar(ct, shrink=0.9, label="Strength [a.u.]")
        axes.set_xlabel("time [s]")
        axes.set_xlim(timeslice)
        plt.tight_layout()
        if save_spec != False:
            plt.savefig(save_spec + ".png", format="png", dpi=300)
        else:
            plt.show()
        plt.close()
    
    dt = np.mean(np.diff(times))
    lags = dt*sig.correlation_lags(len(band1_data), len(band2_data))
    
    cross_corr_coeff = np.fft.ifft(np.fft.fft(ref_data) * np.conj(
            np.fft.fft(ch_data))) / (len(ch_data) * np.sqrt(
                    np.mean(ref_data ** 2) * np.mean(ch_data ** 2)))
    ccs = np.roll(cross_corr_coeff, int(0.5 * len(cross_corr_coeff)))
    
    #ccs = sig.correlate(band1_data, band2_data, method="fft")
    if plot_cc == True:
        figure, axes = plt.subplots(1, 1, sharex=True, figsize=(10, 4))
        axes.plot(lags, ccs, "k", linewidth=0.5)
        axes.set_xlim([lags[0], lags[-1]])
        axes.set_xlabel("Lag [s]")
        axes.set_ylabel("Cross Correlation [a.u.]")
        axes.set_title(str(np.round(band1_freq/1000,1)) + "-" + str(np.round(band2_freq/1000,1)) + " kHz cross correlation in t = " + str(list(np.round(timeslice,2))) + " s")
        axes.text(.99, .975, "\#" + str(shot), ha='right', va='top', transform=axes.transAxes)
        if save_cc != False:
            plt.savefig(save_cc + ".pdf", format="pdf", bbox_inches="tight")
        else:
            plt.show()
        plt.close()
    
    return lags, ccs

# Look more clearly for peaks in frequency space
def plot_freq_profile(shot, bes_time, fluct_data, apdpos, col, timeslices, use_ks, fmin=0.0, ylims=None, fmax=None, plot=False, save=False):
    f_arr, k_arr, kf_matrix = kf_spec_sum_windows(bes_time, fluct_data, apdpos, col, timeslices)
    if fmax == None:
        fmax = np.max(f_arr)
    f_start = (np.abs(f_arr-fmin)).argmin()
    f_end = (np.abs(f_arr-fmax)).argmin()
    k_start = (np.abs(k_arr-use_ks[0])).argmin()
    k_end = (np.abs(k_arr-use_ks[-1])).argmin() + 1
    
    f_data = np.sum(kf_matrix[f_start:f_end,k_start:k_end], axis=1)
    
    """base_start = (np.abs(f_arr[f_start:f_end] - 15.0e3)).argmin()
    base_end = (np.abs(f_arr[f_start:f_end] - 25.0e3)).argmin()
    baseline = np.mean(kf_matrix[baseline_start:baseline_end,k_start:k_end])
    f_data = f_data-baseline"""
        
    if plot != False:
        fig, ax = plt.subplots(1, 1, figsize=(7, 4))
        ax.plot(f_arr[f_start:f_end]*1e-3, np.log(np.abs(smooth(f_data,10))**2), linewidth=0.5)
        ax.set_title(plot)
        ax.text(.99, 1.045, "\#" + str(shot), ha='right', va='top', transform=ax.transAxes)
        ax.text(.99, 0.955, r"$k=$" + str(use_ks) + " $\mathrm{m^{-1}}$", ha='right', va='top', transform=ax.transAxes)
        ax.set_xlabel("Frequency [kHz]")
        ax.set_ylabel(r"$\log\vert S(f)\vert^2$")
        ax.set_xlim([f_arr[f_start]*1e-3, f_arr[f_end]*1e-3])
        if ylims != None:
            ax.set_ylim(ylims) # Useful if making multiple related plots - show on same scale
        if save != False:
            plt.savefig(save + ".png", format="png", bbox_inches="tight", dpi=300)
        else:
            plt.show()
        plt.close()
    return f_arr[f_start:f_end], f_data

# Look more clearly for peaks in frequency space
def plot_wavenum_profile(shot, bes_time, fluct_data, apdpos, col, timeslices, f_range, plot=False, fit=None, save=False):
    f_arr, k_arr, kf_matrix = kf_spec_sum_windows(bes_time, fluct_data, apdpos, col, timeslices)
    f_start = (np.abs(f_arr-f_range[0])).argmin()
    f_end = (np.abs(f_arr-f_range[1])).argmin()
    profile = np.sum(kf_matrix[f_start:f_end,:],axis=0)
    
    if fit != None:
        popt, pcov = curve_fit(fit, k_arr[3:10], profile[3:10])
        residuals = profile[3:10] - fit(k_arr[3:10], *popt)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((profile[3:10] - np.mean(profile[3:10]))**2)
        r_squared = 1 - (ss_res / ss_tot)
        fit_x = np.linspace(k_arr[3], k_arr[9], 100)
        fit_y = fit(fit_x, *popt)

    if plot != False:
        fig, ax = plt.subplots(1, 1, figsize=(7, 4))
        #ax.scatter(k_arr, np.log(np.abs(profile)**2), linewidth=0.5, marker="x")
        ax.scatter(k_arr, profile, linewidth=0.5, marker="x")
        if fit != None:
            #ax.plot(fit_x, np.log(np.abs(fit_y)**2), linewidth=0.5)
            ax.plot(fit_x, fit_y, linewidth=0.5)
            annot = r"$\mu=$" + str(np.round(popt[1],3)) + " $\mathrm{m^{-1}}$, $R^2=$" + str(np.round(r_squared,3))
            ax.legend(["Data", fit.__name__ + " fit"], loc="upper right")
            ax.text(0.01, 0.975, annot, ha='left', va='top', transform=ax.transAxes)
        
        ax.set_title(plot)
        ax.text(.99, 1.045, "\#" + str(shot), ha='right', va='top', transform=ax.transAxes)
        ax.set_xlabel(r"Wavenumber [$\mathrm{m^{-1}}$]")
        ax.set_ylabel(r"$S(k)$")
        ax.set_xlim([k_arr[0], k_arr[-1]])
        f_annot = str(list(np.round((np.asarray(f_range)*1e-3).astype(int),0))) + " kHz"
        ax.text(0.01, 0.915, r"$f\in$" + f_annot, ha='left', va='top', transform=ax.transAxes)

        if save != False:
            plt.savefig(save + ".png", format="png", bbox_inches="tight", dpi=300)
        else:
            plt.show()
        plt.close()
    if fit != None:
        return k_arr, profile, popt[1]
    else:
        return k_arr, profile