import numpy as np

def midpoint_displacement(iterations, P, D):
    # Initialize n based on the number of iterations
    n = 2 ** iterations + 1
    
    # Initialize the meshmap with NaNs
    meshmap = np.full((n, n), np.nan)
    
    # Initialize the random map with Gaussian noise
    rand_map = np.random.randn(n, n)
    
    # Set the corners of the meshmap and rand_map to 0
    meshmap[0, 0] = rand_map[0, 0] = 0
    meshmap[0, -1] = rand_map[0, -1] = 0
    meshmap[-1, 0] = rand_map[-1, 0] = 0
    meshmap[-1, -1] = rand_map[-1, -1] = 0
    
    # Initialize a non-random map (all ones)
    fixed_map = np.ones((n, n))
    
    for i in range(iterations):
        
        # Number of divisions at this iteration
        divs = 2 ** i

        # Set scale for randomness in this iteration
        H = 2 - D
        hval = 0.5 ** (H / 2)
        mid_scale = hval ** (i * 2) # randomness for midpoints
        side_scale = hval ** ((i * 2) + 1) # randmoness for sides

        # Size of a square on this iteration
        sqsz = (n - 1) / divs
        for xi in range(divs):
            x_min = int(sqsz * xi)
            x_max = int(sqsz * (xi + 1))
            x_mid = int((x_max - x_min) / 2 + x_min)

            for yi in range(divs):
                y_min = int(sqsz * yi)
                y_max = int(sqsz * (yi + 1))
                y_mid = int((y_max - y_min) / 2 + y_min)

                # Center point
                meshmap[x_mid, y_mid] = (
                    mid_scale * ((1 - P) * fixed_map[x_mid, y_mid])
                    + mid_scale * (P * rand_map[x_mid, y_mid])
                    + (meshmap[x_min, y_min] + meshmap[x_max, y_min] + meshmap[x_min, y_max] + meshmap[x_max, y_max]) / 4
                )

                # Top mid
                if y_min == 0:
                    meshmap[x_mid, y_min] = (
                        side_scale * ((1 - P) * fixed_map[x_mid, y_min])
                        + side_scale * (P * rand_map[x_mid, y_min])
                        + (meshmap[x_min, y_min] + meshmap[x_max, y_min] + meshmap[x_mid, y_mid]) / 3
                    )
                else:
                    meshmap[x_mid, y_min] = (
                        side_scale * ((1 - P) * fixed_map[x_mid, y_min])
                        + side_scale * (P * rand_map[x_mid, y_min])
                        + (meshmap[x_min, y_min] + meshmap[x_max, y_min] + meshmap[x_mid, y_mid] + meshmap[x_mid, int(y_min - sqsz / 2)]) / 4
                    )

                # Right mid
                if x_max == n-1:
                    meshmap[x_max, y_mid] = (
                        side_scale * ((1 - P) * fixed_map[x_max, y_mid])
                        + side_scale * (P * rand_map[x_max, y_mid])
                        + (meshmap[x_max, y_min] + meshmap[x_max, y_max] + meshmap[x_mid, y_mid]) / 3
                    )
                else:
                    meshmap[x_max, y_mid] = (
                        side_scale * ((1 - P) * fixed_map[x_max, y_mid])
                        + side_scale * (P * rand_map[x_max, y_mid])
                        + (meshmap[x_max, y_min] + meshmap[x_max, y_max] + meshmap[x_mid, y_mid] + meshmap[int(x_max + sqsz / 2), y_mid]) / 4
                    )

                # Bottom mid
                if y_max == n-1:
                    meshmap[x_mid, y_max] = (
                        side_scale * ((1 - P) * fixed_map[x_mid, y_max])
                        + side_scale * (P * rand_map[x_mid, y_max])
                        + (meshmap[x_min, y_max] + meshmap[x_max, y_max] + meshmap[x_mid, y_mid]) / 3
                    )
                else:
                    meshmap[x_mid, y_max] = (
                        side_scale * ((1 - P) * fixed_map[x_mid, y_max])
                        + side_scale * (P * rand_map[x_mid, y_max])
                        + (meshmap[x_min, y_max] + meshmap[x_max, y_max] + meshmap[x_mid, y_mid] + meshmap[x_mid, int(y_mid + sqsz / 2)]) / 4
                    )

                # Left mid
                if x_min == 0:
                    meshmap[x_min, y_mid] = (
                        side_scale * ((1 - P) * fixed_map[x_min, y_mid])
                        + side_scale * (P * rand_map[x_min, y_mid])
                        + (meshmap[x_min, y_min] + meshmap[x_min, y_max] + meshmap[x_mid, y_mid]) / 3
                    )
                else:
                    meshmap[x_min, y_mid] = (
                        side_scale * ((1 - P) * fixed_map[x_min, y_mid])
                        + side_scale * (P * rand_map[x_min, y_mid])
                        + (meshmap[x_min, y_min] + meshmap[x_min, y_max] + meshmap[x_mid, y_mid] + meshmap[int(x_min - sqsz / 2), y_mid]) / 4
                    )

    return meshmap