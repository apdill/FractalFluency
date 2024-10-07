import numpy as np

def prepare_scalemap(scalemap):
    
    scalemap = (scalemap - np.min(scalemap)) / (np.max(scalemap) - np.min(scalemap))

    
    n = scalemap.shape[0]
    
    # Normalize scalemap so minimum value is 0
    scalemap = scalemap - np.min(scalemap)
    
    # Calculate scaling factors
    scalefactor1 = (n-2) / np.max(scalemap)
    scalemap = scalemap * scalefactor1 + 1
    
    # Calculate second scaling factor and apply
    scalefactor2 = (n-1) / (np.max(scalemap) - np.min(scalemap))
    scalemap = np.round(scalemap * scalefactor2)
    
    return scalemap