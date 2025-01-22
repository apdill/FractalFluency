import warnings
warnings.filterwarnings('ignore')

import os
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import gc
import glob
from FF.fractal_generation import branching_network as bn
from fracstack import portfolio_plot
import logging
import sys
from datetime import datetime

# Set up logging
logging.basicConfig(
    filename='random_position_experiment.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def run_random_position_experiment(n_neurons, 
                                   n_positions, 
                                   analysis_params, 
                                   neuron_params, 
                                   network_params, 
                                   plot=True, 
                                   save_path=None):
    """
    Run an experiment where neurons are randomly positioned and measure overlap metrics.
    
    Args:
        n_positions (int): Number of different random positions to try
        n_neurons (int): Number of neurons to simulate
        analysis_params (dict): Parameters for analysis (min_size, max_size, etc.)
        neuron_params (dict): Parameters for neuron generation
        network_params (dict): Parameters for network generation
        plot (bool): Whether to plot results
        save_path (str): Path to save results
        
    Returns:
        tuple: Lists containing mask_info and overlap_info for each position
    """
    try:
        logging.info(f"Starting experiment with {n_neurons} neurons, {n_positions} positions")
        
        if save_path is not None:
            neuron_specific_path = os.path.join(save_path, f'{n_neurons}_neurons')
            os.makedirs(neuron_specific_path, exist_ok=True)

            portfolio_save_dir = os.path.join(neuron_specific_path, 'scaling_plots')
            os.makedirs(portfolio_save_dir, exist_ok=True)

        min_size = analysis_params['min_size']
        max_size = analysis_params['max_size']
        num_sizes = analysis_params['num_sizes']
        num_pos = analysis_params['num_pos']

        # Generate the network once
        net = bn.generate_network(network_id='random_net', 
                                neuron_params=neuron_params, 
                                network_params=network_params)

        fit_info_list = []
        overlap_info_list = []

        # Run for different random positions
        for i in tqdm(range(n_positions), desc=f'{n_neurons} neurons', leave=False):
            try:
                logging.info(f"Processing position {i+1}/{n_positions}")
                
                overlap_metrics_dict = net.scramble_network()

                if overlap_metrics_dict is not False:     
                    if save_path:
                        portfolio_plot_fname = f'[{n_neurons}]neurons_position[{i}].png'
                    else:
                        portfolio_plot_fname = None
                else:
                    logging.warning(f'Failed to generate network for {n_neurons} neurons in position {i}')
                    continue

                # Analyze current configuration
                fit_info_dict = portfolio_plot(net.outline_mask, min_size, max_size, num_sizes, num_pos, (21,7), 
                                    save_dir=portfolio_save_dir if save_path else None,
                                    f_name=portfolio_plot_fname)
                
                overlap_info_list.append(overlap_metrics_dict)
                fit_info_list.append(fit_info_dict)

                # Periodic memory cleanup
                if i % 10 == 0:
                    plt.close('all')
                    gc.collect()
                
                # Force stdout flush to keep progress bar updating
                sys.stdout.flush()
                
            except Exception as e:
                logging.error(f"Error in position {i}: {str(e)}")
                continue

        if plot:
            
            plt.close('all')
            plt.switch_backend('Agg')

            outline_D0_list = [info['D0'] for info in fit_info_list]
            outline_R2_D0_list = [info['R2_D0'] for info in fit_info_list]
            outline_D1_list = [info['D1'] for info in fit_info_list]
            outline_R2_D1_list = [info['R2_D1'] for info in fit_info_list]

            overlap_areas = [info['overlap_area'] for info in overlap_info_list]
            overlap_percentages = [info['overlap_percentage'] for info in overlap_info_list]
            filled_overlap_percentages = [info['filled_overlap_percentage'] for info in overlap_info_list]
            mean_soma_distance_list = [info['mean_soma_distance'] for info in overlap_info_list]
            std_soma_distance_list = [info['std_soma_distance'] for info in overlap_info_list]
            intersections_per_neuron_list = [info['intersections_per_neuron'] for info in overlap_info_list]
            std_intersections_per_neuron_list = [info['std_intersections_per_neuron'] for info in overlap_info_list]
            total_intersections_list = [info['total_intersections'] for info in overlap_info_list]

            plot_configs = [
                {
                    'x_data': overlap_percentages,
                    'y_data': [outline_D0_list, outline_R2_D0_list, outline_D1_list, outline_R2_D1_list],
                    'x_label': 'Overlap Percentage',
                    'y_labels': ['Fractal Dimension (D)', 'R2', 'D1', 'R2'],
                    'titles': ['D vs. Overlap Percentage', 'R2 vs. Overlap Percentage',
                              'D1 vs. Overlap Percentage', 'R2 vs. Overlap Percentage'],
                    'f_name': 'overlap_percentage_metrics.png'
                },
                {
                    'x_data': overlap_areas,
                    'y_data': [outline_D0_list, outline_R2_D0_list, outline_D1_list, outline_R2_D1_list],
                    'x_label': 'Overlap Area',
                    'y_labels': ['Fractal Dimension (D)', 'R2', 'D1', 'R2'],
                    'titles': ['D vs. Overlap Area', 'R2 vs. Overlap Area',
                              'D1 vs. Overlap Area', 'R2 vs. Overlap Area'],
                    'f_name': 'overlap_area_metrics.png'
                },
                {
                    'x_data': filled_overlap_percentages,
                    'y_data': [outline_D0_list, outline_R2_D0_list, outline_D1_list, outline_R2_D1_list],
                    'x_label': 'Filled Overlap Percentage',
                    'y_labels': ['Fractal Dimension (D)', 'R2', 'D1', 'R2'],
                    'titles': ['D vs. Filled Overlap Percentage', 'R2 vs. Filled Overlap Percentage',
                              'D1 vs. Filled Overlap Percentage', 'R2 vs. Filled Overlap Percentage'],
                    'f_name': 'filled_overlap_percentage_metrics.png'
                },

                {
                    'x_data': mean_soma_distance_list,
                    'y_data': [outline_D0_list, outline_R2_D0_list, outline_D1_list, outline_R2_D1_list],
                    'x_label': 'Mean Soma Distance',
                    'y_labels': ['Fractal Dimension (D)', 'R2', 'D1', 'R2'],
                    'titles': ['D vs. Mean Soma Distance', 'R2 vs. Mean Soma Distance',
                              'D1 vs. Mean Soma Distance', 'R2 vs. Mean Soma Distance'],
                    'f_name': 'mean_soma_distance_metrics.png'
                },

                {
                    'x_data': std_soma_distance_list,
                    'y_data': [outline_D0_list, outline_R2_D0_list, outline_D1_list, outline_R2_D1_list],
                    'x_label': 'Standard Deviation of Soma Distance',
                    'y_labels': ['Fractal Dimension (D)', 'R2', 'D1', 'R2'],
                    'titles': ['D vs. Standard Deviation of Soma Distance', 'R2 vs. Standard Deviation of Soma Distance',
                              'D1 vs. Standard Deviation of Soma Distance', 'R2 vs. Standard Deviation of Soma Distance'],
                    'f_name': 'std_soma_distance_metrics.png'
                },


                {
                    'x_data': intersections_per_neuron_list,
                    'y_data': [outline_D0_list, outline_R2_D0_list, outline_D1_list, outline_R2_D1_list],
                    'x_label': 'Intersections per Neuron',
                    'y_labels': ['Fractal Dimension (D)', 'R2', 'D1', 'R2'],
                    'titles': ['D vs. Intersections per Neuron', 'R2 vs. Intersections per Neuron',
                              'D1 vs. Intersections per Neuron', 'R2 vs. Intersections per Neuron'],
                    'f_name': 'intersections_per_neuron_metrics.png'
                },

                {
                    'x_data': std_intersections_per_neuron_list,
                    'y_data': [outline_D0_list, outline_R2_D0_list, outline_D1_list, outline_R2_D1_list],
                    'x_label': 'Standard Deviation of Intersections per Neuron',
                    'y_labels': ['Fractal Dimension (D)', 'R2', 'D1', 'R2'],
                    'titles': ['D vs. Standard Deviation of Intersections per Neuron', 'R2 vs. Standard Deviation of Intersections per Neuron',
                              'D1 vs. Standard Deviation of Intersections per Neuron', 'R2 vs. Standard Deviation of Intersections per Neuron'],
                    'f_name': 'std_intersections_per_neuron_metrics.png'
                },

                {
                    'x_data': total_intersections_list,
                    'y_data': [outline_D0_list, outline_R2_D0_list, outline_D1_list, outline_R2_D1_list],
                    'x_label': 'Total Intersections',
                    'y_labels': ['Fractal Dimension (D)', 'R2', 'D1', 'R2'],
                    'titles': ['D vs. Total Intersections', 'R2 vs. Total Intersections',
                              'D1 vs. Total Intersections', 'R2 vs. Total Intersections'],
                    'f_name': 'total_intersections_metrics.png'
                }

            ]

            for config in plot_configs:

                fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
                axes = [ax1, ax2, ax3, ax4]
                
                for ax, y_data, title, y_label in zip(axes, config['y_data'], config['titles'], config['y_labels']):
                    ax.scatter(config['x_data'], y_data, alpha=0.6)
                    ax.set_xlabel(config['x_label'])
                    ax.set_ylabel(y_label)
                    ax.set_title(title)
                
                plt.tight_layout()
                
                if save_path:
                    plt.savefig(os.path.join(neuron_specific_path, config['f_name']))
                plt.close('all')
        
        return fit_info_list, overlap_info_list
        
    except Exception as e:
        logging.error(f"Fatal error in experiment: {str(e)}")
        raise

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

def main():

    save_path = r'/home/apd/Projects/FractalFluency/code/experiments/random_position_experiment/random_positions'
    
    n_positions = 100
    n_neurons_list = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    
    for n_neurons in tqdm(n_neurons_list, desc=f'Processing neurons'):
        try:
            process_single_neuron_case(n_neurons, n_positions, save_path)
            plt.close('all')
            gc.collect()
        except Exception as e:
            print(f"Error processing n_neurons={n_neurons}: {str(e)}")
            continue
    
    # After all processing is complete, combine all results
    combine_all_results(save_path)

def process_single_neuron_case(n_neurons, n_positions, save_path):
    try:
        start_time = datetime.now()
        logging.info(f"Starting processing for {n_neurons} neurons at {start_time}")
        
        neuron_specific_results = []
        
        total_length = neuron_params['total_length']
        network_params = {
            'width': 2048 + total_length*2,
            'height': 2048 + total_length*2,
            'num_neurons': n_neurons,
            'seed_coordinates': None,
            'edge_margin': total_length + 200,
        }   
        
        fit_info_list, overlap_info_list = run_random_position_experiment(
            n_neurons, n_positions, analysis_params, neuron_params, network_params, 
            plot=True, save_path=save_path
        )
        
        # Combine the data for each position
        for pos, (fit_info, overlap_info) in enumerate(zip(fit_info_list, overlap_info_list)):
            result = {
                'n_neurons': n_neurons,
                'position': pos,
                'D0': fit_info['D0'],
                'R2_D0': fit_info['R2_D0'],
                'D1': fit_info['D1'],
                'R2_D1': fit_info['R2_D1'],
                'overlap_area': overlap_info['overlap_area'],
                'overlap_percentage': overlap_info['overlap_percentage'],
                'filled_overlap_percentage': overlap_info['filled_overlap_percentage'],
                'mean_soma_distance': overlap_info['mean_soma_distance'],
                'std_soma_distance': overlap_info['std_soma_distance'],
                'total_intersections': overlap_info['total_intersections'],
                'intersections_per_neuron': overlap_info['intersections_per_neuron'],
                'std_intersections_per_neuron': overlap_info['std_intersections_per_neuron']
            }
            neuron_specific_results.append(result)
            
        # Create DataFrame and save results
        neuron_results_df = pd.DataFrame(neuron_specific_results)
        
        if save_path:
            neuron_specific_path = os.path.join(save_path, f'{n_neurons}_neurons')
            os.makedirs(neuron_specific_path, exist_ok=True)
            results_path = os.path.join(neuron_specific_path, f'{n_neurons}_neurons_results.csv')
            neuron_results_df.to_csv(results_path, index=False)
            print(f"Saved results for {n_neurons} neurons to {results_path}")

        end_time = datetime.now()
        duration = end_time - start_time
        logging.info(f"Completed {n_neurons} neurons in {duration}")
        
        return neuron_results_df
        
    except Exception as e:
        logging.error(f"Error processing {n_neurons} neurons: {str(e)}")
        raise

def combine_all_results(save_path):
    """Combine all individual neuron results into one master DataFrame"""
    print("Combining all results into one file...")
    
    # Find all results CSV files
    all_csv_files = glob.glob(os.path.join(save_path, "*_neurons/*_results.csv"))
    
    if not all_csv_files:
        print("No result files found!")
        return
    
    # Read and combine all CSV files
    all_dfs = []
    for csv_file in all_csv_files:
        df = pd.read_csv(csv_file)
        all_dfs.append(df)
    
    # Concatenate all DataFrames
    master_df = pd.concat(all_dfs, ignore_index=True)
    
    # Sort by n_neurons and position
    master_df = master_df.sort_values(['n_neurons', 'position'])
    
    # Save the combined results
    master_csv_path = os.path.join(save_path, 'combined_results.csv')
    master_df.to_csv(master_csv_path, index=False)
    print(f"Combined results saved to: {master_csv_path}")
    
    return master_df

if __name__ == "__main__":
    main()
