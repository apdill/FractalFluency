import numpy as np
from PIL import Image
import os

def save_as_tif(input_array, save_path, f_name):
    norm_array = ((input_array - input_array.min()) / np.ptp(input_array) * 255).astype(np.uint8)
    if norm_array.ndim == 3 and norm_array.shape[2] == 1:
        norm_array = np.squeeze(norm_array, axis=2)

    img = Image.fromarray((255-norm_array), mode='L')
    tiff_file = os.path.join(save_path, f"{os.path.splitext(f_name)[0]}_processed.tif")
    
    img.save(tiff_file, format='TIFF')