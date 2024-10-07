import os
import csv

from .process_image_to_array import process_image_to_array
from .measure_D import measure_D

def batch_box_counter(base_path):
    
    results = []
    
    for f_name in os.listdir(base_path):
        im_path = os.path.join(base_path, f_name)
        
        # Skip directories and non-image files
        if os.path.isdir(im_path) or not f_name.lower().endswith(('.jpg','.png', '.jpeg', '.tiff', '.bmp', '.tif')):
            continue
        
        input_array = process_image_to_array(im_path, threshold=150)
        
        d_value = measure_D(input_array, min_size = 8, max_size = 1000, n_sizes = 20, invert = True)
        
        print(f"D-value for {f_name}: {d_value:.3f}")
        
        output_csv_path = os.path.join(base_path, 'D_analysis.csv')
        
        with open(output_csv_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Filename', 'D-value'])
            writer.writerows(results)

    print(f"Results saved to {output_csv_path}")