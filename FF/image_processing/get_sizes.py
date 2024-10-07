import numpy as np

def get_sizes(num_sizes, minsize, maxsize):
    sizes = list(np.around(np.geomspace(minsize, maxsize, num_sizes)).astype(int))
    for index in range(1, len(sizes)):
        size = sizes[index]
        prev_size = sizes[index - 1]
        if size <= prev_size:
            sizes[index] = prev_size + 1
            if prev_size == maxsize:
                return sizes[:index]
    return sizes