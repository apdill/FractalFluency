import numpy as np

def get_d_value(sizes, counts):
    fit = np.polyfit(np.log10(sizes), np.log10(counts), 1)
    return -fit[0]