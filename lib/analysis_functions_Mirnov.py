from lib.imports_BES import *
from lib.analysis_functions_BES import *

def get_kf_spec_mirnov(mirnov_time, mirnov_data, coilpos_angles, timeslice):    
    space_array = [coilpos_angles[i] for i in range(len(coilpos_angles))] # Get Z-coordinates of each coil
    space_array = np.asarray(space_array)
    spec = []
    for coil in range(len(coilpos_angles)): # FFT each coil in time
        fft = fft_channel(mirnov_time, mirnov_data, coil, timeslice)
        f_transform = fft[1]
        spec.append(f_transform) # Each row of spec corresponds to a coil.
    f_arr = fft[0] # Frequency array is the same for all coils so just save any one.
    
    spec = np.asarray(spec)
    spec = np.transpose(spec) # Now each row is a time point as required by calc_kspec
    calc = calc_kspec(spec, space_array) # Get k-f spectrum
    
    kf_matrix = calc[0] # This contains the transform data
    k_arr = calc[1] # This is the array of wavenumbers
    
    return f_arr, k_arr, kf_matrix

# DEPRECATED: plot_kf_spec_mirnov. Use instead plot_dispersion_relation(...) in analysis_functions_general
# Main function for plotting dispersion relations. Arguments offer control of plot type and saving
def plot_kf_spec_mirnov(f_arr, k_arr, kf_matrix, plot_title, fint=50.0, fmin=0.0, fmax=None, smooth_pts=None, conditional=False, save=False):
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
    plt.xlabel("Poloidal Mode Number")
    if save != False:
        plt.savefig(save + ".png", format="png", bbox_inches="tight", dpi=300)
    else:
        plt.show()
    plt.close()

# DEPRECATED: plot_kf_phases. Use instead plot_dispersion_relation(...) in analysis_functions_general
# Plots a graph like a dispersion relation but instead of power it shows phase.
def plot_kf_phases(f_arr, k_arr, kf_matrix, plot_title, fint=50.0, fmin=0.0, fmax=None, smooth_pts=None, conditional=False, save=False):
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
        cbar_label = "Phase (conditional)"
    else:
        cbar_label = "Phase"
    
    # Only need to plot a section of the spectrum. At least half not needed.
    kf_matrix = kf_matrix[(np.abs(f_arr - fmin)).argmin():(np.abs(f_arr - fmax)).argmin()]
    f_arr = f_arr[(np.abs(f_arr - fmin)).argmin():(np.abs(f_arr - fmax)).argmin()]
    
    # Shift phase relative to a particular point, then shift to right range.
    kf_matrix = np.angle(kf_matrix)
    refphase = kf_matrix[19,20]
    print(f_arr[20], refphase)
    kf_matrix -= refphase
    for f in range(len(f_arr)):
        for m in range(len(k_arr)):
            if kf_matrix[f,m] < -np.pi:
                kf_matrix[f,m] += 2*np.pi
            elif kf_matrix[f,m] > np.pi:
                kf_matrix[f,m] -= 2*np.pi

    # Convert to array with rounded frequencies for easier of plotting.
    kf_matrix_plot = pd.DataFrame(kf_matrix, index=np.around(f_arr*1e-3,0), columns=np.around(k_arr,1))

    # For only plotting tick at every x kHz
    interval = int(np.abs(f_arr - fint).argmin())

    # Plot log of values so small enough range for features to be visible
    sns.heatmap(kf_matrix_plot[::-1], yticklabels=interval, cmap="plasma", cbar_kws={"label": cbar_label})
    plt.title(plot_title)
    plt.ylabel("Frequency [kHz]")
    plt.xlabel("Poloidal Mode Number")
    if save != False:
        plt.savefig(save + ".png", format="png", bbox_inches="tight", dpi=300)
    else:
        plt.show()
    plt.close()
    
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

# Function to approximate closest distance from a given Mirnov coil to the separatrix in a given time
def coil_to_sep(equilib_time, equilib_R, equilib_Z, equilib_psi, coilpos, timeslice, coil):
    # Only looking for points in the relevant Z and towards outside
    # Add on bits to ensure the contour extends far enough
    zmin = (np.abs(equilib_Z+0.5)).argmin()
    zmax = (np.abs(equilib_Z-0.5)).argmin()
    rmin = (np.abs(equilib_R-1.0)).argmin()
    
    # Convert timeslice to indices of equilib_time
    if isinstance(timeslice, list) or isinstance(timeslice, np.ndarray):
        idx1 = np.abs(equilib_time-timeslice[0]).argmin()
        idx2 = np.abs(equilib_time-timeslice[1]).argmin()
    else:
        idx1 = np.abs(equilib_time-timeslice).argmin()
        idx2 = idx1
    
    avgdist = 0.0
    for time in range(idx1, idx2+1): # For each equilibria time point   
        # Get path of separatrix
        equilib_psi_t = equilib_psi[time] # Data at this time
        CS = plt.contour(equilib_R[rmin:], equilib_Z[zmin:zmax], equilib_psi_t[zmin:zmax,rmin:], [1.0])
        plt.close()
        verts = mplPath.Path(CS.allsegs[0][0]).vertices # Convert from path to 2D array of points
        dists = []
        for pt in verts: # For each point on separatrix
            dists.append(np.linalg.norm(coilpos[coil]-pt)) # Find distance to coil
        avgdist += np.min(dists) # Take smallest distance only
    return avgdist/(idx2+1-idx1) # Calculate average