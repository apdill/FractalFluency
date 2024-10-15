import numpy as np
from .boxcount import boxcount
from .scaling_plot import scaling_plot
from .get_d_value import get_d_value

def measure_D(input_array, min_size = 1, max_size = 1000, n_sizes = 20, invert = True, plot_image = False):
    sizes, counts = boxcount(input_array, min_size= min_size, max_size=max_size, num_sizes=n_sizes, invert=invert)
    sizes = np.array(sizes)
    counts = np.array(counts)
    
    d_value = get_d_value(sizes, counts)
    #print(f"D-value: {d_value:.3f}")
    
    if plot_image:
        scaling_plot(sizes, counts)
    
    return d_value