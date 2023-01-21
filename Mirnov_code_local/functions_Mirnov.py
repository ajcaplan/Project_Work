### A selection of functions potentially useful to Mirnov data analysis ###
from imports_Mirnov import *

# Simple plot of coil data
def plot_coil_data(shot, mirnov_time, mirnov_data, coil_num, timeslice=None, savefig=False, show=True):
    if timeslice == None:
        plt.plot(mirnov_time, mirnov_data, "k", linewidth=0.5)
    else:
        idx1 = (np.abs(mirnov_time - timeslice[0])).argmin()
        idx2 = (np.abs(mirnov_time - timeslice[1])).argmin()
        plt.plot(mirnov_time[idx1:idx2], mirnov_data[idx1:idx2], "k", linewidth=0.5)
    plt.title("Shot " + str(shot) + ", Mirnov coil " + str(coil_num))
    plt.xlabel("Time [s]")
    plt.ylabel("Volts [V]")

    if savefig == True:
        if timeslice == None:
            plt.savefig("Plots/Mirnov_coil_" + str(coil_num) + "_alltime_volts.pdf", format="pdf", bbox_inches="tight")
        else:
            plt.savefig("Plots/Mirnov_coil_" + str(coil_num) + "_" + str(timeslice[0]) + "to" + str(timeslice[1]) +  "_volts.pdf", format="pdf", bbox_inches="tight")

    if show == True:
        plt.show()
    plt.close()

# FFT in time data from one coil
def get_coil_fft(mirnov_time, mirnov_data, timeslice=None):
    # timeslice in form [t1,t2]
    # mirnov_time: 1D array of times for one coil
    # mirnov_data: 1D array of data for one coil
    if timeslice != None:
        idx1 = (np.abs(mirnov_time - timeslice[0])).argmin()
        idx2 = (np.abs(mirnov_time - timeslice[1])).argmin()
    else:
        idx1 = 0
        idx2 = len(mirnov_time)
    
    ps_mirnov = np.fft.fft(mirnov_data[idx1:idx2])
    time_step = mirnov_time[idx1+1] - mirnov_time[idx1] # Assumes uniform step?
    freqs = np.fft.fftfreq(mirnov_data[idx1:idx2].size,time_step) # In Hz
    idx_mirnov = np.argsort(freqs)

    return [(freqs[idx_mirnov]), (ps_mirnov[idx_mirnov])]

# Plot log of abs square of FFT
def plot_coil_fft(shot, mirnov_time, mirnov_data, coil_num, timeslice=None, savefig=False, show=True):
    transform = get_coil_fft(mirnov_time, mirnov_data, timeslice)
    plt.plot(transform[0], np.log(np.abs(transform[1])**2), "k", linewidth=0.5)
    plt.title("Shot " + str(shot) + ", Mirnov Coil " + coil_num + " FFT")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Power")

    if show == True:
        plt.show()
    if savefig == True:   
        if timeslice == None:
            plt.savefig("Plots/Mirnov_coil_" + str(coil_num) + "_alltime_FFT.pdf", format="pdf", bbox_inches="tight")
        else:
            plt.savefig("Plots/Mirnov_coil_" + str(coil_num) + "_" + str(timeslice[0]) + "to" + str(timeslice[1]) +  "_FFT.pdf", format="pdf", bbox_inches="tight")
    plt.close()
