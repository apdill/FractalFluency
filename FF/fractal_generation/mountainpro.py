import numpy as np
from numba import njit

@njit
def mountainpro(scalemap, iterations, zslice):
    """
    Optimized function to generate slicexz and slicexzline using Numba.
    
    Parameters:
    scalemap (np.ndarray): 2D NumPy array representing the scalemap.
    iterations (int): Number of iterations used to determine the size.
    zslice (float): Normalized z-slice value (0 to 1).
    
    Returns:
    tuple: Two 2D NumPy arrays (slicexz, slicexzline) of dtype uint8.
    """
    n = 2 ** iterations
    zslice = int(np.ceil(n * zslice))
    
    sliceloop = n // 2  # down the middle
    
    # Initialize slicexz and slicexzline with zeros
    slicexz = np.zeros((n, n), dtype=np.uint8)
    slicexzline = np.zeros((n, n), dtype=np.uint8)
    
    tracexz = scalemap[sliceloop, :]
    
    oldxz = tracexz[0]
    for ct in range(n):
        # Ensure tracexz[ct] does not exceed the bounds of slicexz
        max_tracexz = tracexz[ct]
        if max_tracexz > (n - 1):
            max_tracexz = n - 1
        max_tracexz = int(max_tracexz)
        
        # Set slicexz[ctc, ct] = 1 for ctc in range(max_tracexz)
        for ctc in range(max_tracexz):
            slicexz[ctc, ct] = 1
        
        # Calculate the range for slicexzline
        min_val = oldxz if oldxz < tracexz[ct] else tracexz[ct]
        max_val = oldxz if oldxz > tracexz[ct] else tracexz[ct]
        
        # Ensure the range does not exceed the bounds
        if min_val < 0:
            min_val = 0
        if max_val >= n:
            max_val = n - 1
        min_val = int(min_val)
        max_val = int(max_val) + 1  # Inclusive of max_val
        
        # Set slicexzline[ctc, ct] = 1 for ctc in the calculated range
        for ctc in range(min_val, max_val):
            slicexzline[ctc, ct] = 1
        
        oldxz = tracexz[ct]
    
    return slicexz, slicexzline

'''
import numpy as np

def mountainpro(scalemap, iterations, zslice):
    n = 2 ** iterations
    zslice = int(np.ceil(n * zslice))
    
    sliceloop = n // 2  # down the middle

    slicexz = np.zeros((n, n), dtype=np.uint8)
    slicexzline = np.zeros((n, n), dtype=np.uint8)

    tracexz = scalemap[sliceloop, :]

    oldxz = tracexz[0]
    for ct in range(n):
        # Ensure tracexz[ct] does not exceed the bounds of slicexz
        max_tracexz = min(int(tracexz[ct]), n - 1)
        
        for ctc in range(max_tracexz):
            slicexz[ctc, ct] = 1
        
        for ctc in range(int(min(oldxz, tracexz[ct])), min(int(max(oldxz, tracexz[ct])) + 1, n)):
            slicexzline[ctc, ct] = 1
        
        oldxz = tracexz[ct]
    
    return slicexz, slicexzline

# Example usage:
# max_iter = 9  # Example value for max_iter
# zslice = 0.5  # Example value for zslice (normalized, 0 to 1)
# scalemap = np.random.rand((2 ** max_iter), (2 ** max_iter))  # Example scalemap
# slicexz, slicexzline = mountainpro(max_iter, zslice, scalemap)
'''