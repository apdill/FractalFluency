import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import gc
import warnings
import logging
from datetime import datetime
from FF.fractal_generation import branching_network as bn
from fracstack import portfolio_plot

# Suppress warnings
warnings.filterwarnings('ignore')

# Set up logging with buffering
logging.basicConfig(
    filename='merging_experiment.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    buffering=1024*8
)

# Analysis parameters
analysis_params = {
    'min_size': 16,
    'max_size': 506,
    'num_sizes': 50,
    'num_pos': 100,
}

# Neuron parameters
neuron_params = {
    'depth': 6,
    'mean_soma_radius': 1,
    'std_soma_radius': 1,
    'D': 1.5,
    'branch_angle': np.pi / 3,
    'mean_branches': 1,
    'weave_type': 'Gauss',
    'randomness': 0.2,
    'curviness': 'Gauss',
    'curviness_magnitude': 1,
    'n_primary_dendrites': 6,
    'initial_thickness': 10,
    'total_length': 600,
}

def run_merging_position_experiment(radius, n_neurons, n_steps, analysis_params, neuron_params, network_params, plot=True, save_path=None):
    """
    Run an experiment where neurons move from outer circle towards center and measure overlap metrics.
    """
    try:
        if save_path is not None:
            neuron_specific_path = os.path.join(save_path, f'{n_neurons}_neurons')
            os.makedirs(neuron_specific_path, exist_ok=True)
            portfolio_save_dir = os.path.join(neuron_specific_path, 'scaling_plots')
            os.makedirs(portfolio_save_dir, exist_ok=True)

        logging.info(f"Starting merging experiment with {n_neurons} neurons, {n_steps} steps")
        
        angles = np.linspace(0, 2*np.pi, n_neurons, endpoint=False)
        center_x = network_params['width'] // 2
        center_y = network_params['height'] // 2

        # Generate seed coordinates
        seed_coordinates = []
        for angle in angles:
            x = center_x + radius * np.cos(angle)
            y = center_y + radius * np.sin(angle)
            orientation = angle + np.pi  # Point towards center
            seed_coordinates.append((x, y, orientation))

        network_params['seed_coordinates'] = seed_coordinates
        
        # Generate the network
        net = bn.generate_network(
            network_id='merging_net',
            neuron_params=neuron_params,
            network_params=network_params
        )

        step_size = radius / n_steps
        mask_info_list = []
        overlap_info_list = []

        # Run simulation steps
        for i in tqdm(range(n_steps), desc=f'{n_neurons} neurons'):
            if i % 5 == 0:
                logging.info(f"Processing step {i+1}/{n_steps}")

            # Move each neuron towards center
            for j in range(n_neurons):
                angle = angles[j]
                dx = -step_size * np.cos(angle)
                dy = -step_size * np.sin(angle)
                net.translate_neuron(j, int(dx), int(dy))

            # Generate portfolio plot
            if save_path:
                portfolio_plot_fname = f'[{n_neurons}]neurons_step[{i}].png'
            else:
                portfolio_plot_fname = None

            mask_info = portfolio_plot(
                net.outline_mask,
                analysis_params['min_size'],
                analysis_params['max_size'],
                analysis_params['num_sizes'],
                analysis_params['num_pos'],
                (21,7),
                save_dir=portfolio_save_dir if save_path else None,
                f_name=portfolio_plot_fname
            )

            overlap_metrics_df = net.calculate_overlap_metrics()
            
            mask_info_list.append(mask_info)
            overlap_info_list.append(overlap_metrics_df)

            # Periodic cleanup
            if i % 20 == 0:
                plt.close('all')
                gc.collect()

        return mask_info_list, overlap_info_list

    except Exception as e:
        logging.error(f"Error in experiment: {str(e)}")
        raise

def main():
    save_path = r'/home/apd/Projects/FractalFluency/datasets/fractal_portfolio/branch_networks/overlap_analysis/merging_positions'
    
    radius = 1500
    n_steps = 20
    n_neurons_list = [2, 3, 4, 5, 6]
    all_results = []

    for n_neurons in tqdm(n_neurons_list, desc='Processing neurons'):
        try:
            start_time = datetime.now()
            logging.info(f"Starting processing for {n_neurons} neurons at {start_time}")

            network_params = {
                'width': 5000,
                'height': 5000,
                'num_neurons': n_neurons,
                'seed_coordinates': None,
            }

            mask_info_list, overlap_info_list = run_merging_position_experiment(
                radius, n_neurons, n_steps, analysis_params, neuron_params, network_params,
                plot=True, save_path=save_path
            )

            # Combine the data for each step
            for step, (mask_info, overlap_info) in enumerate(zip(mask_info_list, overlap_info_list)):
                result = {
                    'n_neurons': n_neurons,
                    'step': step,
                    'D0': float(mask_info['D0'].iloc[0]),
                    'R2_D0': float(mask_info['R2_D0'].iloc[0]),
                    'D1': float(mask_info['D1'].iloc[0]),
                    'R2_D1': float(mask_info['R2_D1'].iloc[0]),
                    'overlap_area': overlap_info['overlap_area'],
                    'overlap_percentage': overlap_info['overlap_percentage'],
                    'filled_overlap_percentage': overlap_info['filled_overlap_percentage'],
                    'mean_soma_distance': overlap_info['mean_soma_distance'],
                    'std_soma_distance': overlap_info['std_soma_distance'],
                    'total_intersections': overlap_info['total_intersections'],
                    'intersections_per_neuron': overlap_info['intersections_per_neuron'],
                    'std_intersections_per_neuron': overlap_info['std_intersections_per_neuron']
                }
                all_results.append(result)

            end_time = datetime.now()
            duration = end_time - start_time
            logging.info(f"Completed {n_neurons} neurons in {duration}")

            # Clear memory after each complete neuron case
            plt.close('all')
            gc.collect()

        except Exception as e:
            logging.error(f"Error processing n_neurons={n_neurons}: {str(e)}")
            continue

    # Save all results
    results_df = pd.DataFrame(all_results)
    if save_path:
        results_df.to_csv(os.path.join(save_path, 'merging_positions_results.csv'), index=False)
        logging.info("Saved final results to CSV")

if __name__ == "__main__":
    main() 