import numpy as np
from numba import njit

@njit
def mountainpro(scalemap, iterations, zslice, scale_factor=1.0):
    """
    Optimized function to generate mountain profile slices using Numba.
    
    Parameters:
    scalemap (np.ndarray): 2D NumPy array representing the scalemap.
    iterations (int): Number of iterations used to determine the size.
    zslice (float): Normalized z-slice value (0 to 1).
    scale_factor (float): Factor to scale the profile vertically.
    
    Returns:
    tuple: Two 2D NumPy arrays (profile_filled, profile_outline) of dtype uint8.
    """
    grid_size = 2 ** iterations
    zslice = int(np.ceil(grid_size * zslice))
    
    middle_row = grid_size // 2  # down the middle
    
    # Initialize output arrays with zeros
    profile_filled = np.zeros((grid_size, grid_size), dtype=np.uint8)
    profile_outline = np.zeros((grid_size, grid_size), dtype=np.uint8)
    
    height_values = scalemap[middle_row, :]
    
    # Normalize heights to [0, 1]
    min_height = height_values.min()
    max_height = height_values.max()
    if max_height == min_height:
        max_height = min_height + 1e-9  # Avoid divide-by-zero
    
    normalized_heights = (height_values - min_height) / (max_height - min_height)
    
    # Scale and center the heights
    scaled_heights = (
        (normalized_heights - 0.5) * scale_factor * (grid_size - 1) + (grid_size / 2)
    )
    scaled_heights = np.clip(scaled_heights, 0, grid_size - 1).astype(np.int32)  # Clamp to bounds

    prev_height = scaled_heights[0]

    for x_pos in range(grid_size):
        current_height = scaled_heights[x_pos]
        
        # Fill the profile up to the scaled height
        for y_pos in range(current_height + 1):  # Inclusive of max height
            profile_filled[y_pos, x_pos] = 1
        
        # Outline between scaled previous and current heights
        min_outline = min(prev_height, current_height)
        max_outline = max(prev_height, current_height)
        for y_pos in range(min_outline, max_outline + 1):  # Inclusive of max height
            profile_outline[y_pos, x_pos] = 1
        
        # Update the previous height for the next iteration
        prev_height = current_height
    
    return profile_filled, profile_outline

