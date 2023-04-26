from lib.imports_BES import *

# Work out centre of magnetic system in the middle of a given timeslice
def get_magnetic_centre(equilib_time, equilib_R, equilib_Z, equilib_psi, timeslice):
    # Equilib shouldn't shift much over timeslice so middle is reasonable
    time_idx = (np.abs(equilib_time - (timeslice[0]+timeslice[1])*0.5)).argmin()
    
    min_idx = equilib_psi[time_idx].argmin() # This computes based on flattened array, returning a single int.
    row = min_idx//len(equilib_R)
    col = min_idx - row*len(equilib_R)
    
    # Return real coordinates of minimum
    return equilib_R[col], equilib_Z[row]

# Find the cross correlation between two arrays (of equal size). Returns lags in seconds and coefficients.
def cross_corr(refch, ch, dt):
    cross_corr_coeff = np.fft.ifft(np.fft.fft(refch) * np.conj(
            np.fft.fft(ch))) / (len(ch) * np.sqrt(
                    np.mean(refch ** 2) * np.mean(ch ** 2)))
    lags = dt*np.linspace(-0.5*len(cross_corr_coeff), 0.5*len(cross_corr_coeff), len(cross_corr_coeff))
    return lags, np.roll(cross_corr_coeff, int(0.5 * len(cross_corr_coeff)))

# Work out how far apart a BES channel focal point is from a Mirnov coil
def bes_mirnov_dist(apdpos, bes_ch, coilpos, mirnov_idx):
    path = coilpos[mirnov_idx] - apdpos[bes_ch]
    return np.sqrt(path[0]**2 + path[1]**2), path

# Main function for plotting dispersion relations (magnetic or BES - change xlabel). Arguments offer control of plot type and saving.
def plot_dispersion_relation(f_arr, k_arr, kf_matrix, plot_title, xlabel,
                             fmin=0.0, fmax=None, phases=False, smooth_pts=None, conditional=False, save=False):
    # If no maximum f given as arg, set it as max possible
    if fmax == None:
        fmax = np.max(f_arr)
    
    # If smoothing selected, use function in imports_BES.py
    if smooth_pts != None:
        tmp_matrix = np.transpose(kf_matrix)
        for line in range(len(k_arr)):
            tmp_matrix[line] = smooth(tmp_matrix[line], smooth_pts)
        kf_matrix = np.transpose(tmp_matrix)
    
    # Only need to plot a section of the spectrum. At least f<0 half not needed.    
    kf_matrix = kf_matrix[(np.abs(f_arr - fmin)).argmin():(np.abs(f_arr - fmax)).argmin()]
    f_arr = f_arr[(np.abs(f_arr - fmin)).argmin():(np.abs(f_arr - fmax)).argmin()]
    
    # If selected in args, make probability distribution
    if conditional == True:
        kf_matrix = np.transpose(np.transpose(kf_matrix)/np.sum(kf_matrix,axis=1))
        #cbar_label = r"$\log\big(\vert S(k|f)\vert^2\big)$"
        cbar_label = r"$\log\big(\vert S(k|f)\vert^2\big)$"
    else:
        cbar_label = r"$\log\big(\vert S(f,k)\vert^2\big)$"

    # set bounds to change tick labels on plot 
    bounds = [np.min(k_arr), np.max(k_arr), np.min(f_arr)*1e-3, np.max(f_arr)*1e-3]
    fig, ax = plt.subplots(1,1)
    
    # Plot log of values so small enough range for features to be visible
    if phases == False:
        cs = ax.matshow(np.log(np.abs(kf_matrix)**2), origin="lower", extent=bounds, cmap="plasma", aspect="auto")
    else:
        cs = ax.matshow(np.angle(kf_matrix), origin="lower", extent=bounds, cmap="plasma", aspect="auto")
    fig.colorbar(cs, label=cbar_label, shrink=0.9)
    
    # Adjust graph appearance, add labels
    ax.set_title(plot_title)
    ax.set_ylabel("Frequency [kHz]")
    ax.set_xlabel(xlabel)
    ax.xaxis.set_ticks_position('bottom')
    if save != False:
        plt.savefig(save + ".pdf", format="pdf", bbox_inches="tight", dpi=300)
    else:
        plt.show()
    plt.close()
    
    
# Take dispersion relation and get frequency profile in a given range
def get_f_profile(f_arr, kf_matrix, frange, smooth_pts=5, vlines=None, plot=False):
    fstart = (np.abs(f_arr-frange[0])).argmin()
    fend = (np.abs(f_arr-frange[1])).argmin()
    f_arr = f_arr[fstart:fend+1]
    
    # So this works for dispersion relation or 1D FFT
    if kf_matrix.ndim == 2:
        profile = np.sum(np.abs(kf_matrix)**2, axis=1)
    else:
        profile = np.abs(kf_matrix)**2
    profile = smooth(profile[fstart:fend+1],smooth_pts)
    
    if plot != False:
        fig, ax = plt.subplots(1,1)
        ax.plot(f_arr*1e-3, np.log(profile))
        if vlines != None: # Vertical line checking for area calculations
            ax.vlines(vlines[0]*1e-3, ax.get_ylim()[0], ax.get_ylim()[1], "k", linewidth=0.5)
            ax.vlines(vlines[1]*1e-3, ax.get_ylim()[0], ax.get_ylim()[1], "k", linewidth=0.5)
        ax.set_ylabel("Power [a.u.]")
        ax.set_xlabel("Frequency [kHz]")
        ax.set_title(plot)
        plt.show()
        plt.close()
        
    return f_arr, profile

# From above profile, determine area between frequencies, accounting for noise
def get_f_area(f_arr, f_profile, frange):
    fstart = (np.abs(f_arr-frange[0])).argmin()
    fend = (np.abs(f_arr-frange[1])).argmin()
    full = trapz(f_profile[fstart:fend+1], x=f_arr[fstart:fend+1])
    noise = trapz([f_profile[fstart],p[1][fend]], x=[f_arr[fstart],f_arr[fend]])
    return full-noise