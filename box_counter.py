import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
from sklearn.metrics import r2_score 
Image.MAX_IMAGE_PIXELS = None  # Disable the limit
import imageio


def scaling_plot(sizes, counts, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))
    fit = np.polyfit(np.log10(sizes), np.log10(counts), 1)
    ax.scatter(np.log10(sizes), np.log10(counts), color = 'black')
    ax.plot(np.log10(sizes), np.log10(sizes) * fit[0] + fit[1], color = 'red')
    ax.set_xlabel('Log(L)')
    ax.set_ylabel('Log(N)')

def get_d_value(sizes, counts):
    fit = np.polyfit(np.log10(sizes), np.log10(counts), 1)
    return -fit[0]

def boxcount(array, num_sizes=10, min_size=None, max_size=None, invert=False):
    """
    array: 2D numpy array (counts elements that aren't 0, or elements that aren't 1 if inverted)
    num_sizes: number of box sizes
    min_size: smallest box size in pixels (defaults to 1)
    max_size: largest box size in pixels (defaults to 1/5 smaller dimension of array)
    invert: 1 - array, if you want to count 0s instead of 1s
    """
    if invert:
        array = 1 - array
    min_size = 1 if min_size is None else min_size
    max_size = max(min_size + 1, min(array.shape) // 5) if max_size is None else max_size
    sizes = get_sizes(num_sizes, min_size, max_size)
    counts = []
    for size in sizes:
        counts.append(get_mincount(array, size))
    return sizes, counts

def get_mincount(array, size):
    shape = array.shape
    count = 0
    for i in range(0, shape[0], size):
        for j in range(0, shape[1], size):
            if np.any(array[i:i + size, j:j + size]):
                count += 1
    return count

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

def process_image_to_array(file_path, threshold=None):
    image = Image.open(file_path)
    image_array = np.array(image, dtype=np.uint8)
    
    # Convert to grayscale by averaging channels if it's not already
    if len(image_array.shape) == 3:
        image_array = image_array.mean(axis=2)
    
    # Binarize the image with the given threshold
    if threshold is None:
        threshold = np.max(image_array)
        
    binary_image_array = (image_array >= threshold).astype(np.uint8) 
        
    return binary_image_array

def invert_array(arr):
    return np.where(arr == 0, 1, 0)

def save_as_tif(input_array, save_path, f_name):
    norm_array = (invert_array(input_array) * 255).astype(np.uint8)
    tiff_file = os.path.join(save_path, f"{os.path.splitext(f_name)[0]}.tif")
    imageio.imwrite(tiff_file, invert_array(norm_array))
    
    
def measure_D(input_array, min_size = 1, max_size = 1000, n_sizes = 20, invert = True):
    sizes, counts = boxcount(input_array, min_size= min_size, max_size=max_size, num_sizes=n_sizes, invert=invert)
    sizes = np.array(sizes)
    counts = np.array(counts)
    
    d_value = get_d_value(sizes, counts)
    print(f"D-value: {d_value:.3f}")
    
    scaling_plot(sizes, counts)
    
    return d_value

    
    