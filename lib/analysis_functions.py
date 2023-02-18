from lib.imports_BES import *

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

def get_crash_adjacent_window(utda_time, utda_data, threshold, timeslice, start_shift=5e-3, end_shift=1e-3):
    crash_times = get_crash_times(utda_time, utda_data, threshold, timeslice)
    windows = []
    
    for i in crash_times:
        windows.append([i-start_shift, i-end_shift])
        
    return windows
