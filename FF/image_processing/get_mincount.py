import numpy as np

def get_mincount(array, size):
    shape = array.shape
    count = 0
    for i in range(0, shape[0], size):
        for j in range(0, shape[1], size):
            if np.any(array[i:i + size, j:j + size]):
                count += 1
    return count