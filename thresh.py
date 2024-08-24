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

def thresh(scalemap, medianz=True, zslice_ratio=0.5):
    n = scalemap.shape[0]
    
    if not medianz:
        sortheight = np.sort(scalemap.flatten())  # Sorted list of all height values
        zslice = sortheight[int(zslice_ratio * len(sortheight))]  # Sea level as the black/white ratio
    else:
        zslice = np.median(scalemap)  # Median value of the scalemap
    
    blackcount = 0
    threshmap = np.zeros_like(scalemap, dtype=np.uint8)
    
    for xx in range(n):
        for yy in range(n):
            if scalemap[xx, yy] <= zslice:
                threshmap[xx, yy] = 0  # Black
                blackcount += 1
            else:
                threshmap[xx, yy] = 255  # White
    
    bwratio = blackcount / (n ** 2)
    return threshmap, bwratio

