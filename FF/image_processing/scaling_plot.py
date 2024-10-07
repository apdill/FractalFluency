import numpy as np
import matplotlib.pyplot as plt

def scaling_plot(sizes, counts, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))
    fit = np.polyfit(np.log10(sizes), np.log10(counts), 1)
    ax.scatter(np.log10(sizes), np.log10(counts), color = 'black')
    ax.plot(np.log10(sizes), np.log10(sizes) * fit[0] + fit[1], color = 'red')
    ax.set_xlabel('Log(L)')
    ax.set_ylabel('Log(N)')