### A selection of functions potentially useful to Mirnov data analysis ###
from imports_Mirnov import *

# Convert coil number (e.g. 210) to index (e.g. 3)
def coil_idx(num):
    return np.where(coil_nums == num)[0][0]

# Simple plot of coil data
def plot_coil_data(shot, mirnov_time, mirnov_data, coil_index, timeslice=None, savefig=False, show=True):
    if timeslice == None:
        timeslice = [mirnov_time[0], mirnov_time[-1]]
    idx1 = (np.abs(mirnov_time - timeslice[0])).argmin()
    idx2 = (np.abs(mirnov_time - timeslice[1])).argmin()+1
    
    plt.plot(mirnov_time[idx1:idx2], mirnov_data[coil_index, idx1:idx2], "k", linewidth=0.25)
    plt.title("Shot " + str(shot) + ", Mirnov coil " + str(coil_index))
    plt.xlabel("Time [s]")
    plt.ylabel("Volts [V]")

    if savefig == True:
        plt.savefig("Plots/Mirnov_coil_" + str(coil_index) + "_" + str(timeslice[0]) + "to" + str(timeslice[1]) +  "_volts.pdf", format="pdf", bbox_inches="tight")

    if show == True:
        plt.show()
    plt.close()

# Simple plot of coil positions
def plot_coil_positions(coilpos, coil_nums, save=False):
    plt.scatter(coilpos[:,0], coilpos[:,1], marker="x")
    for i, txt in enumerate(coil_nums):
        plt.annotate(txt, (coilpos[i,0], coilpos[i,1]), xytext=(coilpos[i,0]-0.03, coilpos[i,1]-0.025))

    plt.xlabel("R [m]")
    plt.ylabel("Z [m]")
    plt.title("Mirnov Coil Positions and Numbers")
    plt.xlim(left=1.4,right=1.9)
    if save == True:
        plt.savefig("coil_pos.pdf", format="pdf", bbox_inches="tight")
    else:
        plt.show()
    plt.close()
    
def plot_coils_with_equilib(shot, coilpos, timepoint, equilib_R, equilib_Z, equilib_psi, save=False):
    # plot the location of the BES view array with flux surfaces
    figure, axes = plt.subplots(1, 1, figsize=(7.5, 5.0))
    axes.contour(equilib_R, equilib_Z, equilib_psi, np.linspace(0, 1, 21), 
                 colors='k')
    axes.scatter(coilpos[:, 0], coilpos[:, 1], marker="x")
    axes.set_aspect('equal', adjustable='datalim')
    axes.set_xlabel('radius R [m]', fontsize=18)
    axes.set_ylabel('height above midplane z [m]', fontsize=18)
    axes.set_xlim([0.0, 2.25])
    axes.set_ylim([-1.5, 1.0])
    axes.tick_params(axis='x', labelsize=16)
    axes.tick_params(axis='y', labelsize=16)
    axes.set_title("Shot " + str(shot) + ', time' + str('{:<06}'.format(
        round(timepoint, 4))) + ' s Mirnov and Equilbria', fontsize=22)
    
    if save==True:
        plt.savefig('./Plots/shot' + str(shot) + '_BES_locs_' + 
                str(int(1000*round(timepoint, 3))) +'ms.pdf', 
                format='pdf', transparent=True, bbox_inches="tight")
    else:
        plt.show()
    plt.close()

# FFT in time data from one coil
def get_coil_fft(mirnov_time, mirnov_data, coil_index, timeslice):
    idx1 = (np.abs(mirnov_time - timeslice[0])).argmin()
    idx2 = (np.abs(mirnov_time - timeslice[1])).argmin()
    
    ps_mirnov = np.fft.fft(mirnov_data[coil_index, idx1:idx2])
    time_step = mirnov_time[idx1+1] - mirnov_time[idx1] # Assumes uniform step?
    freqs = np.fft.fftfreq(mirnov_data[coil_index, idx1:idx2].size,time_step) # In Hz
    idx_mirnov = np.argsort(freqs)

    return [(freqs[idx_mirnov]), (ps_mirnov[idx_mirnov])]

# Plot log of abs square of FFT
def plot_coil_fft(shot, mirnov_time, mirnov_data, coil_index, timeslice=None, savefig=False, show=True):
    if timeslice == None:
        timeslice = [mirnov_time[0], mirnov_time[-1]]
    idx1 = (np.abs(mirnov_time - timeslice[0])).argmin()
    idx2 = (np.abs(mirnov_time - timeslice[1])).argmin()
    
    transform = get_coil_fft(mirnov_time, mirnov_data, coil_index, timeslice=timeslice)
    plt.plot(transform[0]*1e-3, np.log(np.abs(transform[1])**2), "k", linewidth=0.5)
    plt.title("Shot " + str(shot) + ", Mirnov Coil " + str(coil_index) + " FFT, t=" + str(timeslice) + " s")
    plt.xlabel("Frequency [kHz]")
    plt.ylabel(r"$\log\vert S(f)\vert^2$")
    

    if show == True:
        plt.show()
    if savefig == True:   
        if timeslice == None:
            timeslice = [mirnov_time[0], mirnov_time[-1]]
        plt.savefig("Plots/Mirnov_coil_" + str(coil_index) + "_" + str(timeslice[0]) + "to" + str(timeslice[1]) +  "_FFT.pdf", format="pdf", bbox_inches="tight")
    plt.close()
