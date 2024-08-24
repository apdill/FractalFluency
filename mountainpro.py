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
