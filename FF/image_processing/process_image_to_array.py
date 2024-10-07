import numpy as np
from PIL import Image

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