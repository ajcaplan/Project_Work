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