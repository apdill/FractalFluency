import os
import numpy as np
import imageio
import pandas as pd
from thresh import thresh, prepare_scalemap
from edgedetector import edgedetector
from mountainpro import mountainpro
from midpoint_displacement import midpoint_displacement

def batch_generate_fractals(num_fractals, iterations, D_range, P, 
                            fractal_type='coastline edge', 
                            output_dir='fractals_batch'):
    """
    Generates multiple fractal images with varying D values and saves them in the specified directory.

    :param num_fractals: Number of fractals to generate.
    :param iterations: Number of iterations for the midpoint displacement algorithm.
    :param D_range: Tuple (min_D, max_D) defining the range of D values.
    :param P: The parameter controlling the randomness in the midpoint displacement.
    :param fractal_type: The type of fractal to generate ('coastline edge', 'greyscale', 'coastline BW').
    :param output_dir: Directory where the generated fractal images will be saved.
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    min_D, max_D = D_range
    D_values = np.linspace(min_D, max_D, num_fractals)

    data = []

    for i, D in enumerate(D_values):
        # Generate the fractal pattern
        meshmap = midpoint_displacement(iterations, P, D)
        scalemap = prepare_scalemap(meshmap)
        threshmap, _ = thresh(scalemap, iterations)

        # Determine the fractal generation function based on the type
        fractal_map = {
            'coastline edge': lambda: edgedetector(threshmap, iterations),
            'greyscale': lambda: meshmap,
            'coastline BW': lambda: threshmap
        }

        # Validate fractal type
        if fractal_type not in fractal_map:
            print('Invalid fractal type chosen, aborting.')
            return

        fractal = fractal_map[fractal_type]()
        
        # Define the filename and save the fractal as a TIFF image
        tiff_file = os.path.join(output_dir, f"fractal_{i}.tif")
        imageio.imwrite(tiff_file, fractal)
        print(f'Generated fractal {i+1}/{num_fractals} with Iterations={iterations}, P={P:.2f}, D={D:.2f}')

        # Add the filename and D value to the list
        data.append([f"fractal_{i}.tif", D])

    # Save the filenames and D values to a CSV file
    csv_file = os.path.join(output_dir, 'labels.csv')
    df = pd.DataFrame(data, columns=['filename', 'd_value'])
    df.to_csv(csv_file, index=False)

    print(f'Batch generation complete. {num_fractals} fractals saved to {output_dir}.')
    print(f'Labels saved to {csv_file}.')

# Example usage:
# batch_generate_fractals(100, iterations=8, D_range=(1.1, 1.7), P=0.5)
