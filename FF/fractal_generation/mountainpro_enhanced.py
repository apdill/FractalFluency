import numpy as np

def mountainpro_enhanced(scalemap, iterations, zslice=0.5, num_slices=5, random_slices=False, random_state=None):
    """
    Generates mountain profiles by aggregating multiple central or random slices.

    Parameters:
        scalemap (np.ndarray): The processed scalemap.
        iterations (int): Number of iterations used in fractal generation.
        zslice (float): Normalized slice position (0 to 1) for central slice selection.
        num_slices (int): Number of slices to aggregate.
        random_slices (bool): If True, selects random slices instead of central consecutive slices.
        random_state (int, optional): Seed for random number generator for reproducibility.

    Returns:
        slicexz (np.ndarray): Binary image representing the mountain.
        slicexzline (np.ndarray): Binary image representing the mountain profile lines.
    """
    n = 2 ** iterations
    slicexz = np.zeros((n, n), dtype=np.uint8)
    slicexzline = np.zeros((n, n), dtype=np.uint8)
    
    # Clamp zslice to [0, 1]
    zslice = min(max(zslice, 0.0), 1.0)
    
    # Initialize random number generator if random slices are requested
    rng = np.random.default_rng(seed=random_state) if random_slices else None
    
    # Determine slice indices based on selection mode
    if random_slices:
        # Ensure num_slices does not exceed total number of slices

        if num_slices > n:
            raise ValueError(f"num_slices ({num_slices}) cannot exceed total number of slices ({n}).")
        
        # Select unique random slices across the entire scalemap
        slice_indices = rng.choice(n, size=num_slices, replace=False)
    else:
        # Calculate slice_start and slice_end ensuring they are integers
        slice_start = int(round(n * zslice - num_slices / 2))
        slice_end = slice_start + num_slices
        
        # Ensure slice_start and slice_end are within bounds
        slice_start = max(slice_start, 0)
        slice_end = min(slice_end, n)
        
        # Generate a list of consecutive slice indices
        slice_indices = np.arange(slice_start, slice_end)
        
        # If the number of slices is less than requested (due to boundary), adjust
        actual_num_slices = slice_indices.size
        if actual_num_slices < num_slices:
            print(f"Warning: Requested {num_slices} slices, but only {actual_num_slices} slices are available within bounds.")

 # Iterate over the selected slices
    for sliceno in slice_indices:
        tracexz = scalemap[sliceno, :]
        oldxz = tracexz[0]
        
        for ct in range(n):
            # Ensure tracexz[ct] does not exceed the bounds of slicexz
            max_tracexz = min(int(tracexz[ct]), n - 1)
            
            for ctc in range(max_tracexz):
                slicexz[ctc, ct] = 1
            
            # Define error range based on previous value
            min_xz = min(int(oldxz), int(tracexz[ct]))
            max_xz = max(int(oldxz), int(tracexz[ct])) + 1
            
            for ctc in range(min_xz, min(max_xz, n)):
                slicexzline[ctc, ct] = 1
            
            oldxz = tracexz[ct]
    
    return slicexz, slicexzline