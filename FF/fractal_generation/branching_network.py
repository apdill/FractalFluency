import numpy as np
import cv2
import random
from scipy.interpolate import splprep, splev 
from skimage.filters import sobel 
import pandas as pd
from skimage import draw

def generate_network(network_params, neuron_params, network_id = None):
    """
    Generates a network with the given ID and saves outputs to the specified directory structure.

    Args:
        network_id (str): Unique identifier for the network.
        output_base_dir (str): Base directory for outputs.
        neuron_params (dict): Parameters for neuron creation.
        network_params (dict): Parameters for network creation (e.g., width, height, num_neurons).
    """
    network_width = network_params.get('width', 2048)
    network_height = network_params.get('height', 2048)
    num_neurons = network_params.get('num_neurons', 10)
    edge_margin = network_params.get('edge_margin', 100)
    seed_coordinates = network_params.get('seed_coordinates', None)
    deterministic = network_params.get('deterministic', False)

    # Create and generate the network
    network = Network(width=network_width, 
                      height=network_height, 
                      num_neurons=num_neurons, 
                      neuron_params=neuron_params, 
                      edge_margin=edge_margin,
                      seed_coordinates=seed_coordinates,
                      network_id=network_id,
                      deterministic=deterministic)
    

    network.seed_neurons()
    network.grow_network()

    return network



def compute_max_buffer_distance(width, height, num_neurons, mean_soma_radius, std_soma_radius, edge_margin):
    """
    Computes the maximum buffer distance that allows num_neurons to fit within the network
    given the mean_soma_radius, std_soma_radius, and fixed edge_margin.
    """
    # Define the maximum soma radius
    max_soma_radius = mean_soma_radius + 2 * std_soma_radius

    # Effective dimensions
    effective_width = width - 2 * edge_margin
    effective_height = height - 2 * edge_margin

    # Set initial search bounds for buffer_distance
    low = 0  # Minimum buffer distance
    # Maximum possible buffer_distance
    high = min(effective_width, effective_height) - 2 * max_soma_radius
    high = max(high, 0)  # Ensure non-negative

    max_buffer_distance = 0

    while low <= high:
        mid = (low + high) // 1  # Current buffer distance to test

        # Calculate min_soma_distance with current buffer_distance
        min_soma_distance = 2 * max_soma_radius + mid

        # Calculate number of cells along each dimension
        num_cells_x = int(effective_width // min_soma_distance)
        num_cells_y = int(effective_height // min_soma_distance)
        total_cells = num_cells_x * num_cells_y

        if num_cells_x > 0 and num_cells_y > 0 and total_cells >= num_neurons:
            # Buffer distance is acceptable, try increasing it
            max_buffer_distance = mid
            low = mid + 1
        else:
            # Buffer distance is too large, decrease it
            high = mid - 1

    return int(max_buffer_distance)



class Soma:
    """
    Represents the soma (cell body) of a neuron.
    """
    def __init__(self, position, mean_radius, std_radius):
        self.position = position
        self.radius = max(np.random.normal(mean_radius, std_radius), 0)
        self._generate_soma()

    def _generate_soma(self):
        """Generate the soma coordinates."""
        theta = np.linspace(0, 2 * np.pi, 100)

        # Some random variations
        sine_variation = np.random.uniform(0, 15) * np.sin(2 * theta)
        gaussian_variation = np.random.normal(0, 2, len(theta))
        ellipse_ratio = np.random.uniform(0.8, 1.2)
        elongation_angle = np.random.uniform(0, 2 * np.pi)

        # Parametric equations for x_soma, y_soma
        self.x_soma = (
            (self.radius + gaussian_variation + sine_variation)
            * (
                np.cos(theta) * np.cos(elongation_angle)
                - np.sin(theta) * np.sin(elongation_angle) * ellipse_ratio
            )
            + self.position[0]
        )
        self.y_soma = (
            (self.radius + gaussian_variation + sine_variation)
            * (
                np.sin(theta) * np.cos(elongation_angle)
                + np.cos(theta) * np.sin(elongation_angle) * ellipse_ratio
            )
            + self.position[1]
        )

    def create_binary_masks(self, size, pad_size=0):
        """
        Creates both filled and skeleton binary masks of the soma using stored coordinates.

        Args:
            size (tuple): The size of the masks, e.g. (height, width).
            pad_size (int): Amount of padding to add around the mask.

        Returns:
            dict: A dictionary containing 'filled' and 'skeleton' masks.
        """
        # Ensure x_soma, y_soma are arrays (they should be)
        if isinstance(self.x_soma, np.ndarray):
            x_padded = self.x_soma + pad_size
            y_padded = self.y_soma + pad_size
        else:
            x_padded = self.x_soma
            y_padded = self.y_soma

        # Create an (N, 2) integer array:
        coordinates = np.column_stack((x_padded, y_padded)).astype(np.int32)
        
        # If there's any chance your arrays could be empty, check here:
        # if len(coordinates) == 0:
        #     return {'filled': np.zeros(size, dtype=np.uint8),
        #             'skeleton': np.zeros(size, dtype=np.uint8)}

        # Create the filled mask
        mask_filled = np.zeros(size, dtype=np.uint8)
        cv2.fillPoly(mask_filled, [coordinates], 1)

        # Create the skeleton (outline) mask
        mask_skeleton = np.zeros(size, dtype=np.uint8)
        cv2.polylines(mask_skeleton, [coordinates], isClosed=True, color=1, thickness=1)

        return {'filled': mask_filled, 'skeleton': mask_skeleton}


class Dendrite:
    """
    Represents the dendritic tree of a neuron.
    """

    def __init__(
        self,
        soma,
        depth,
        D,
        branch_angle,
        mean_branches,
        weave_type=None,
        randomness=0.0,
        curviness=None,
        curviness_magnitude=1.0,
        n_primary_dendrites=4,
        total_length=500,
        initial_thickness=10,
        deterministic=False
    ):
        self.soma = soma
        self.depth = depth
        self.D = D
        self.branch_angle = branch_angle
        self.mean_branches = mean_branches
        self.weave_type = weave_type
        self.randomness = randomness
        self.curviness = curviness
        self.curviness_magnitude = curviness_magnitude
        self.n_primary_dendrites = n_primary_dendrites
        self.total_length = total_length
        self.initial_thickness = initial_thickness
        self.branch_lengths = self._generate_branch_lengths()
        self.deterministic = deterministic
        self.num_branches = int(mean_branches) if deterministic else mean_branches

    def _generate_branch_lengths(self):
        r = self.mean_branches ** (-1 / self.D)
        branch_lengths = np.zeros(self.depth)
        normalization_factor = self.total_length / sum(r ** i for i in range(self.depth))

        for i in range(self.depth):
            branch_lengths[i] = normalization_factor * r ** i

        return branch_lengths

    def _calculate_thickness(self, distance_from_start, segment_length):
        proportion_start = 1 - (distance_from_start / self.total_length)
        proportion_end = 1 - ((distance_from_start + segment_length) / self.total_length)

        proportion_start = np.clip(proportion_start, 0, 1)
        proportion_end = np.clip(proportion_end, 0, 1)

        thickness_at_start = self.initial_thickness * (proportion_start) ** (1 / self.D)
        thickness_at_end = self.initial_thickness * (proportion_end) ** (1 / self.D)

        thickness_at_start = max(thickness_at_start, 1)
        thickness_at_end = max(thickness_at_end, 1)

        return thickness_at_start, thickness_at_end
    

    def intra_branch_weave(self, x1, y1, x2, y2, length):
        if self.deterministic:
            return np.array([x1, x2]), np.array([y1, y2])
        
        num_points = max(int(self.curviness_magnitude * 10), 2)
        num_control_points = np.random.randint(4, 7)

        t_values = np.linspace(0.2, 0.8, num_control_points - 2)
        t_values = np.concatenate(([0], t_values, [1]))

        dx = x2 - x1
        dy = y2 - y1
        branch_angle = np.arctan2(dy, dx)
        angle = branch_angle + np.pi / 2

        base_x = x1 + t_values * dx
        base_y = y1 + t_values * dy

        radius = length * np.random.uniform(-0.05, 0.05, size=len(t_values))
        perturb_x = base_x + radius * np.cos(angle)
        perturb_y = base_y + radius * np.sin(angle)

        control_x = perturb_x
        control_y = perturb_y

        # Create a spline through the control points
        tck, u = splprep([control_x, control_y], s=0)
        u_fine = np.linspace(0, 1, num_points)
        xs, ys = splev(u_fine, tck)

        perturbation_scale = length / 200
        if self.curviness == 'Gauss':
            xs += np.random.normal(0, perturbation_scale, num_points)
            ys += np.random.normal(0, perturbation_scale, num_points)
        elif self.curviness == 'Uniform':
            xs += np.random.uniform(-perturbation_scale, perturbation_scale, num_points)
            ys += np.random.uniform(-perturbation_scale, perturbation_scale, num_points)

        xs[0], ys[0] = x1, y1
        xs[-1], ys[-1] = x2, y2

        return xs, ys


    
    def _grow_branch(self, x, y, angle, remaining_depth):
        if remaining_depth == 0:
            return None, []

        branch_length = self.branch_lengths[self.depth - remaining_depth]
        sum_length = sum(self.branch_lengths[: self.depth - remaining_depth])

        thickness_start, thickness_end = self._calculate_thickness(sum_length, branch_length)

        if not self.deterministic:
            if self.weave_type == 'Gauss':
                branch_length *= 1 + np.random.normal(0, self.randomness)
                angle += np.random.normal(0, self.randomness)
            elif self.weave_type == 'Uniform':
                branch_length *= 1 + np.random.uniform(-self.randomness, self.randomness)
                angle += np.random.uniform(-self.randomness, self.randomness)

        end_x = x + branch_length * np.cos(angle)
        end_y = y + branch_length * np.sin(angle)

        weave_x, weave_y = self.intra_branch_weave(x, y, end_x, end_y, branch_length)

        branch_data = {
            'points': np.array([weave_x, weave_y]),
            'length': branch_length,
            'depth': self.depth - remaining_depth,
            'thickness_start': thickness_start,
            'thickness_end': thickness_end,
        }

        num_branches = self.num_branches if self.deterministic else int(
            np.clip(np.round(np.random.normal(self.mean_branches, 1)), 1, None)
        )
        
        new_branches = []
        for i in range(num_branches):
            new_angle = angle + self.branch_angle * (i - (num_branches - 1) / 2)
            if not self.deterministic:
                if self.weave_type == 'Gauss':
                    new_angle += np.random.normal(0, self.randomness)
                elif self.weave_type == 'Uniform':
                    new_angle += np.random.uniform(-self.randomness, self.randomness)
            new_branches.append(((end_x, end_y), new_angle))

        return branch_data, new_branches



class Neuron:
    """
    Represents a neuron, consisting of a soma and dendrites.
    """

    def __init__(
        self,
        position,
        starting_angle = 0,
        depth=5,
        mean_soma_radius=10,
        std_soma_radius=2,
        D=2.0,
        branch_angle=np.pi / 4,
        mean_branches=3,
        weave_type='Gauss',
        randomness=0.1,
        curviness='Gauss',
        curviness_magnitude=1.0,
        n_primary_dendrites=4,
        network=None,
        neuron_id=None,
        initial_thickness = 10,
        total_length = 500,
        n_intersections = None
    ):
        self.network = network
        self.position = position
        self.starting_angle = starting_angle
        self.soma = Soma(position, mean_soma_radius, std_soma_radius)
        self.dendrite = Dendrite(
            self.soma,
            depth,
            D,
            branch_angle,
            mean_branches,
            weave_type,
            randomness,
            curviness,
            curviness_magnitude,
            n_primary_dendrites,
            total_length,
            initial_thickness,
            self.network.deterministic if self.network else False
        )

        self.neuron_id = neuron_id
        self.current_depth = 0
        self.branch_ends = []
        self.is_growing = True
        self.deterministic = self.network.deterministic if self.network else False
        self.branch_data = []  # Add this to store branch information

    def generate_start_points(self):
        """
        Generates the starting points for the primary dendrites and initializes branch ends.
        """
        x_soma = self.soma.x_soma
        y_soma = self.soma.y_soma
        num_soma_points = len(x_soma)

        # Generate evenly spaced angles around the soma (0 to 2π)
        angles = np.linspace(0, 2*np.pi, self.dendrite.n_primary_dendrites, endpoint=False)
        angles = (angles + self.starting_angle) % (2*np.pi)

        if not self.deterministic:
            # Add random rotation and noise only in non-deterministic mode
            random_rotation = np.random.uniform(0, 2*np.pi)
            angles = (angles + random_rotation) % (2*np.pi)
            angle_noise = np.random.uniform(-np.pi/6, np.pi/6, size=len(angles))
            angles += angle_noise

        # Find the closest soma points for each angle
        start_points = []
        for angle in angles:
            # Calculate vector from soma center to perimeter
            dx = x_soma - self.position[0]
            dy = y_soma - self.position[1]
            point_angles = np.arctan2(dy, dx)
            
            # Find the closest point on the soma perimeter
            angle_diff = np.abs((point_angles - angle + np.pi) % (2*np.pi) - np.pi)
            closest_idx = np.argmin(angle_diff)
            start_points.append((x_soma[closest_idx], y_soma[closest_idx]))

        # Initialize branch ends with the calculated points and angles
        self.branch_ends = [(point, angle) for point, angle in zip(start_points, angles)]


    def prepare_next_layer(self):
        """
        Prepare the proposed branches for the next layer without updating the dendrite mask.

        Returns:
            list: Proposed branches for the next layer.
        """
        if self.current_depth >= self.dendrite.depth or not self.branch_ends:
            self.is_growing = False
            return []

        proposed_branches = []

        for start_point, angle in self.branch_ends:
            branch_data, new_branches = self.dendrite._grow_branch(
                start_point[0], start_point[1], angle, self.dendrite.depth - self.current_depth
            )

            if branch_data is not None:
                proposed_branches.append(
                    {
                        'branch_data': branch_data,
                        'start_point': start_point,
                        'new_branches': new_branches,  # Include new branch ends
                    }
                )

        return proposed_branches

    def add_branches(self, accepted_branches, network_mask_filled, network_mask_skeleton):
        """
        Add the accepted branches directly to the network masks and update branch ends.
        """
        new_branch_ends = []

        for branch_info in accepted_branches:
            branch_data = branch_info['branch_data']
            points = branch_data['points']
            new_branches = branch_info['new_branches']

            # Store branch data for later mask generation
            self.branch_data.append(branch_data)

            coordinates = np.column_stack((points[0], points[1])).astype(np.int32)
            
            # Calculate thickness variation
            thickness_start = int(branch_data['thickness_start'])
            thickness_end = int(branch_data['thickness_end'])
            num_segments = len(coordinates) - 1
            thickness_values = np.linspace(thickness_start, thickness_end, num_segments).astype(int)

            for i in range(num_segments):
                cv2.line(
                    network_mask_filled,
                    tuple(coordinates[i]),
                    tuple(coordinates[i + 1]),
                    1,
                    thickness=thickness_values[i]
                )
                # Draw skeleton
                cv2.line(
                    network_mask_skeleton,
                    tuple(coordinates[i]),
                    tuple(coordinates[i + 1]),
                    1,
                    thickness=1  # Fixed skeleton thickness
                )

            del branch_data
            new_branch_ends.extend(new_branches)

        self.branch_ends = new_branch_ends

    def generate_binary_mask(self):
        """
        Generates binary masks for this individual neuron.
        
        Returns:
            dict: A dictionary containing 'filled', 'skeleton', and 'outline' masks for this neuron.
        """
        # Calculate padding needed based on total branch length
        pad_size = int(self.dendrite.total_length * 1.5)  # Add extra padding for safety
        
        # Create padded size
        padded_size = (
            self.network.height + 2 * pad_size,
            self.network.width + 2 * pad_size
        )
        
        # Initialize empty masks with padding
        neuron_mask_filled = np.zeros(padded_size, dtype=np.uint8)
        neuron_mask_skeleton = np.zeros(padded_size, dtype=np.uint8)
        
        # Add soma to masks with padding
        soma_masks = self.soma.create_binary_masks(padded_size, pad_size)
        neuron_mask_filled = np.logical_or(neuron_mask_filled, soma_masks['filled']).astype(np.uint8)
        neuron_mask_skeleton = np.logical_or(neuron_mask_skeleton, soma_masks['skeleton']).astype(np.uint8)
        
        # Draw all stored branches with adjusted coordinates
        for branch_data in self.branch_data:
            points = branch_data['points'].copy()  # Make a copy to avoid modifying original
            points[0] += pad_size  # Adjust x coordinates
            points[1] += pad_size  # Adjust y coordinates
            coordinates = np.column_stack((points[0], points[1])).astype(np.int32)
            
            thickness_start = int(branch_data['thickness_start'])
            thickness_end = int(branch_data['thickness_end'])
            num_segments = len(coordinates) - 1
            thickness_values = np.linspace(thickness_start, thickness_end, num_segments).astype(int)
            
            for i in range(num_segments):
                cv2.line(
                    neuron_mask_filled,
                    tuple(coordinates[i]),
                    tuple(coordinates[i + 1]),
                    1,
                    thickness=thickness_values[i]
                )
                cv2.line(
                    neuron_mask_skeleton,
                    tuple(coordinates[i]),
                    tuple(coordinates[i + 1]),
                    1,
                    thickness=1
                )
        
        # Generate outline using sobel filter
        neuron_mask_outline = sobel(neuron_mask_filled)
        
        return {
            'filled': neuron_mask_filled,
            'skeleton': neuron_mask_skeleton,
            'outline': neuron_mask_outline,
            'pad_size': pad_size
        }



class Network:
    """
    Represents a network of neurons.
    """

    def __init__(self, width, height, num_neurons, neuron_params, edge_margin=100, seed_coordinates=None, network_id=None, deterministic=False):
        self.width = width
        self.height = height
        self.num_neurons = num_neurons
        self.neuron_params = neuron_params
        self.neurons = []
        self.filled_mask = np.zeros((self.height, self.width), dtype=np.uint8)
        self.skeleton_mask = np.zeros((self.height, self.width), dtype=np.uint8)
        self.outline_mask = np.zeros((self.height, self.width), dtype=np.uint8)
        self.neuron_masks = []
        self.network_id = network_id
        self.edge_margin = edge_margin
        self.seed_coordinates = seed_coordinates
        self.deterministic = deterministic

        # Define neuron size parameters
        mean_soma_radius = neuron_params['mean_soma_radius']
        std_soma_radius = neuron_params['std_soma_radius']

        # Compute max_soma_radius
        max_soma_radius = mean_soma_radius + 2 * std_soma_radius

        # Compute max buffer distance
        self.buffer_distance = compute_max_buffer_distance(
            width=self.width,
            height=self.height,
            num_neurons=self.num_neurons,
            mean_soma_radius=mean_soma_radius,
            std_soma_radius=std_soma_radius,
            edge_margin=self.edge_margin
        )

        # Now, with buffer_distance, compute min_soma_distance
        self.min_soma_distance = 2 * max_soma_radius + self.buffer_distance

    def seed_neurons(self):
        """
        Seeds neurons and draws somas directly onto the network mask.
        """

        if self.seed_coordinates is not None:
            # Verify the number of coordinates matches number of neurons
            assert len(self.seed_coordinates) == self.num_neurons, \
                f"Number of manual coordinates ({len(self.seed_coordinates)}) must match number of neurons ({self.num_neurons})"
            
            # Initialize masks if not already initialized
            if not hasattr(self, 'filled_mask'):
                self.filled_mask = np.zeros((self.height, self.width), dtype=np.uint8)
            if not hasattr(self, 'skeleton_mask'):
                self.skeleton_mask = np.zeros((self.height, self.width), dtype=np.uint8)
            if not hasattr(self, 'outline_mask'):
                self.outline_mask = np.zeros((self.height, self.width), dtype=np.uint8)

            # Place neurons at specified coordinates
            for neuron_index, position in enumerate(self.seed_coordinates):
                neuron_id = f"{self.network_id}_neuron_{neuron_index + 1}"
                neuron = Neuron(
                    position = position,
                    starting_angle = position[2] if len(position) == 3 else 0,
                    **self.neuron_params,
                    network=self,
                    neuron_id=neuron_id
                )
                
                soma_masks = neuron.soma.create_binary_masks(size=(self.height, self.width))
                self.filled_mask = np.logical_or(self.filled_mask, soma_masks['filled']).astype(np.uint8)
                self.skeleton_mask = np.logical_or(self.skeleton_mask, soma_masks['skeleton']).astype(np.uint8)
                
                self.neurons.append(neuron)
                neuron.generate_start_points()
            
            return
        
        cell_size = self.min_soma_distance

        # Effective dimensions considering the edge margin
        effective_width = self.width - 2 * self.edge_margin
        effective_height = self.height - 2 * self.edge_margin

        num_cells_x = int(effective_width // cell_size)
        num_cells_y = int(effective_height // cell_size)
        total_cells = num_cells_x * num_cells_y

        if self.num_neurons > total_cells:
            raise ValueError("Cannot place all neurons with the given parameters. Adjust neuron size or number.")

        # Generate all possible grid cell indices within the safe area
        grid_indices = [(i, j) for i in range(num_cells_x) for j in range(num_cells_y)]
        # Randomly select cells to place neurons
        selected_cells = random.sample(grid_indices, self.num_neurons)

        # Initialize the shared network mask if not already initialized
        if not hasattr(self, 'filled_mask'):
            self.filled_mask = np.zeros((self.height, self.width), dtype=np.uint8)
        if not hasattr(self, 'skeleton_mask'):
            self.skeleton_mask = np.zeros((self.height, self.width), dtype=np.uint8)
        if not hasattr(self, 'outline_mask'):
            self.outline_mask = np.zeros((self.height, self.width), dtype=np.uint8)

        for neuron_index, (i, j) in enumerate(selected_cells):
            neuron_id = f"{self.network_id}_neuron_{neuron_index + 1}"
            cell_x = i * cell_size + self.edge_margin
            cell_y = j * cell_size + self.edge_margin

            # Random jitter within the cell, ensuring the neuron stays within bounds
            soma_radius = max(
                self.neuron_params['mean_soma_radius'] + 2 * self.neuron_params['std_soma_radius'], 1
            )
            max_jitter_x = min(cell_size - 2 * soma_radius, effective_width - i * cell_size - 2 * soma_radius)
            max_jitter_y = min(cell_size - 2 * soma_radius, effective_height - j * cell_size - 2 * soma_radius)
            if max_jitter_x < 0 or max_jitter_y < 0:
                raise ValueError("Cell size is too small for the soma size. Adjust neuron parameters.")

            neuron_x = cell_x + soma_radius + random.uniform(0, max_jitter_x)
            neuron_y = cell_y + soma_radius + random.uniform(0, max_jitter_y)
            position = (neuron_x, neuron_y)

            neuron = Neuron(
                position,
                **self.neuron_params,
                network=self,
                neuron_id=neuron_id
            )

            soma_masks = neuron.soma.create_binary_masks(size=(self.height, self.width))

            self.filled_mask = np.logical_or(self.filled_mask, soma_masks['filled']).astype(np.uint8)
            self.skeleton_mask = np.logical_or(self.skeleton_mask, soma_masks['skeleton']).astype(np.uint8)

            # Add neuron to the network
            self.neurons.append(neuron)

            # Generate start points for dendrite growth
            neuron.generate_start_points()

    def grow_single_neuron(self, neuron_index):
        """
        Grows a single neuron in the network to its full depth.
        
        Args:
            neuron_index (int): Index of the neuron to grow
        """
        neuron = self.neurons[neuron_index]
        while neuron.is_growing:
            proposed_branches = neuron.prepare_next_layer()
            if proposed_branches:
                neuron.add_branches(proposed_branches, self.filled_mask, self.skeleton_mask)
                neuron.current_depth += 1
            else:
                neuron.is_growing = False

    def grow_network(self):
        """
        Grows the dendrites of all neurons in the network layer by layer.
        """
        growing = True
        while growing:
            growing = False
            for neuron in self.neurons:
                if neuron.is_growing:
                    proposed_branches = neuron.prepare_next_layer()
                    if proposed_branches:
                        neuron.add_branches(proposed_branches, self.filled_mask, self.skeleton_mask)
                        growing = True
                    else:
                        neuron.is_growing = False

            # Increment the current depth for all neurons
            for neuron in self.neurons:
                if neuron.is_growing:
                    neuron.current_depth += 1
        
        # Update outline mask after growth is complete
        self.outline_mask = (sobel(self.filled_mask) > 0).astype(np.uint8)
        self.store_neuron_masks()


    def store_neuron_masks(self):
        """Store binary masks for each neuron"""
        self.neuron_masks = []
        for neuron in self.neurons:
            # Store original position
            orig_pos = neuron.position
            orig_soma_pos = (neuron.soma.x_soma, neuron.soma.y_soma)
            
            # Generate full masks
            masks = neuron.generate_binary_mask()
            self.neuron_masks.append({
                'filled': masks['filled'],
                'skeleton': masks['skeleton'],
                'outline': masks['outline'],
                'pad_size': masks['pad_size'],
                'original_position': orig_pos,
                'original_soma_position': orig_soma_pos  # Store original soma position too
                
            })


    def translate_neuron(self, neuron_index, dx, dy, rotation=0):
        """
        Rotates the neuron (branches + soma outline) around its OLD center 
        by `rotation` RADIANS, then translates everything by (dx, dy).
        """
        neuron = self.neurons[neuron_index]

        # -- If you want to treat `rotation` as degrees, do the conversion here:
        # theta = np.radians(rotation)
        # For RADIANS, just do this:
        theta = rotation

        cos_t = np.cos(theta)
        sin_t = np.sin(theta)

        # Old neuron center, used as pivot.
        # (Often neuron.position == neuron.soma.position, but check your code.)
        pivot_x, pivot_y = neuron.position

        # ----------------------------------------------------
        # 1) Rotate & translate the soma outline (x_soma, y_soma)
        # ----------------------------------------------------
        # Shift so pivot is at origin:
        x_shifted = neuron.soma.x_soma - pivot_x
        y_shifted = neuron.soma.y_soma - pivot_y

        # Rotate in 2D:
        x_rot = x_shifted * cos_t - y_shifted * sin_t
        y_rot = x_shifted * sin_t + y_shifted * cos_t

        # Shift back + apply final translation:
        x_final = x_rot + pivot_x + dx
        y_final = y_rot + pivot_y + dy

        # Update the soma outline
        neuron.soma.x_soma = x_final
        neuron.soma.y_soma = y_final

        # Update the soma's "position" (center).  
        # For a circle, you might use mean or the pivot itself:
        new_soma_x = pivot_x + dx
        new_soma_y = pivot_y + dy
        neuron.soma.position = (new_soma_x, new_soma_y)

        # ----------------------------------------------------
        # 2) Rotate & translate the branches
        # ----------------------------------------------------
        for branch in neuron.branch_data:
            x_coords = branch['points'][0]
            y_coords = branch['points'][1]

            # Shift to pivot
            x_shifted = x_coords - pivot_x
            y_shifted = y_coords - pivot_y

            # Rotate
            x_rot = x_shifted * cos_t - y_shifted * sin_t
            y_rot = x_shifted * sin_t + y_shifted * cos_t

            # Shift back + apply final translation
            x_final = x_rot + pivot_x + dx
            y_final = y_rot + pivot_y + dy

            # Cast to int32 if you need integer branch coords:
            branch['points'][0] = x_final.astype(np.int32)
            branch['points'][1] = y_final.astype(np.int32)

        # ----------------------------------------------------
        # 3) Update the neuron's "global" position 
        #    (often same as soma.position)
        # ----------------------------------------------------
        neuron.position = (new_soma_x, new_soma_y)

        # ----------------------------------------------------
        # 4) Redraw
        # ----------------------------------------------------
        self.update_network_masks()

    def update_network_masks(self, updated_neuron_index=None):
        """
        Redraw the entire network from scratch using individual neuron data.
        """
        # Clear existing masks
        self.filled_mask = np.zeros((self.height, self.width), dtype=np.uint8)
        self.skeleton_mask = np.zeros((self.height, self.width), dtype=np.uint8)
        
        # Redraw each neuron
        for neuron in self.neurons:
            # Draw soma
            soma_masks = neuron.soma.create_binary_masks(size=(self.height, self.width))
            self.filled_mask = np.logical_or(self.filled_mask, soma_masks['filled']).astype(np.uint8)
            self.skeleton_mask = np.logical_or(self.skeleton_mask, soma_masks['skeleton']).astype(np.uint8)
            
            # Draw branches
            for branch_data in neuron.branch_data:
                points = branch_data['points']
                coordinates = np.column_stack((points[0], points[1])).astype(np.int32)
                
                thickness_start = int(branch_data['thickness_start'])
                thickness_end = int(branch_data['thickness_end'])
                num_segments = len(coordinates) - 1
                thickness_values = np.linspace(thickness_start, thickness_end, num_segments).astype(int)
                
                for i in range(num_segments):
                    cv2.line(
                        self.filled_mask,
                        tuple(coordinates[i]),
                        tuple(coordinates[i + 1]),
                        1,
                        thickness=thickness_values[i]
                    )
                    cv2.line(
                        self.skeleton_mask,
                        tuple(coordinates[i]),
                        tuple(coordinates[i + 1]),
                        1,
                        thickness=1
                    )
        
        # Update outline mask
        self.outline_mask = (sobel(self.filled_mask) > 0).astype(np.uint8)

    def calculate_overlap_metrics(self):
        """
        Calculate overlap and intersection metrics across all neurons in the network.
        """
        if self.num_neurons < 2:
            return {
                'overlap_area': 0,
                'total_area': 0,
                'overlap_percentage': 0,
                'filled_overlap_percentage': 0,
                'mean_soma_distance': 0,
                'std_soma_distance': 0,
                'total_intersections': 0,
                'intersections_per_neuron': 0,
                'std_intersections_per_neuron': 0
            }
        
        # Calculate coverage count
        coverage_count = np.zeros_like(self.filled_mask, dtype=int)
        total_neuron_area = 0
        
        # Create individual masks for each neuron to calculate overlaps
        individual_masks = []
        skeleton_masks = []
        for neuron in self.neurons:
            # Create empty mask
            neuron_mask = np.zeros_like(self.filled_mask)
            skeleton_mask = np.zeros_like(self.skeleton_mask)
            
            # Draw soma
            soma_masks = neuron.soma.create_binary_masks(size=(self.height, self.width))
            neuron_mask = np.logical_or(neuron_mask, soma_masks['filled']).astype(np.uint8)
            skeleton_mask = np.logical_or(skeleton_mask, soma_masks['skeleton']).astype(np.uint8)
            
            # Draw branches
            for branch_data in neuron.branch_data:
                points = branch_data['points']
                coordinates = np.column_stack((points[0], points[1])).astype(np.int32)
                
                thickness_start = int(branch_data['thickness_start'])
                thickness_end = int(branch_data['thickness_end'])
                num_segments = len(coordinates) - 1
                thickness_values = np.linspace(thickness_start, thickness_end, num_segments).astype(int)
                
                for i in range(num_segments):
                    cv2.line(
                        neuron_mask,
                        tuple(coordinates[i]),
                        tuple(coordinates[i + 1]),
                        1,
                        thickness=thickness_values[i]
                    )
                    cv2.line(
                        skeleton_mask,
                        tuple(coordinates[i]),
                        tuple(coordinates[i + 1]),
                        1,
                        thickness=1
                    )
            
            individual_masks.append(neuron_mask)
            skeleton_masks.append(skeleton_mask)
            coverage_count += neuron_mask > 0
            total_neuron_area += np.sum(neuron_mask > 0)
        
        # Calculate overlaps
        overlap_mask = coverage_count > 1
        total_overlap_area = np.sum(overlap_mask)
        total_filled_area = np.sum(self.filled_mask)
        
        overlap_percentage = (total_overlap_area / total_neuron_area * 100) if total_neuron_area > 0 else 0
        filled_overlap_percentage = (total_overlap_area / total_filled_area * 100) if total_filled_area > 0 else 0
        
        # Calculate soma distances
        soma_positions = np.array([neuron.position for neuron in self.neurons])
        squared_distances = []
        for i in range(len(soma_positions)):
            for j in range(i + 1, len(soma_positions)):
                squared_dist = np.sum((soma_positions[i][:2] - soma_positions[j][:2]) ** 2)
                squared_distances.append(squared_dist)
        
        distances = np.sqrt(np.array(squared_distances)) if squared_distances else [0]
        mean_soma_distance = np.mean(distances)
        std_soma_distance = np.std(distances) if len(distances) > 1 else 0
        
        # Calculate intersections using skeleton masks
        neuron_intersections = np.zeros(len(self.neurons))
        total_intersections = 0
        
        for i in range(len(self.neurons)):
            for j in range(i + 1, len(self.neurons)):
                intersection_points = np.logical_and(skeleton_masks[i], skeleton_masks[j])
                num_intersections = np.sum(intersection_points)
                total_intersections += num_intersections
                
                # Add intersections to both neurons' counts
                neuron_intersections[i] += num_intersections
                neuron_intersections[j] += num_intersections
                
                # Update n_intersections attribute for both neurons
                self.neurons[i].n_intersections = neuron_intersections[i]
                self.neurons[j].n_intersections = neuron_intersections[j]
        
        intersections_per_neuron = total_intersections / self.num_neurons if self.num_neurons > 0 else 0
        std_intersections_per_neuron = np.std(neuron_intersections) if len(neuron_intersections) > 0 else 0
        
        return {
            'overlap_area': total_overlap_area,
            'total_area': total_filled_area,
            'overlap_percentage': overlap_percentage,
            'filled_overlap_percentage': filled_overlap_percentage,
            'mean_soma_distance': mean_soma_distance,
            'std_soma_distance': std_soma_distance,
            'total_intersections': int(total_intersections),
            'intersections_per_neuron': float(intersections_per_neuron),
            'std_intersections_per_neuron': float(std_intersections_per_neuron),
        }
    
    def calculate_pairwise_overlap_metrics(self):
        """Calculate pairwise overlap metrics between all neurons in the network"""
        if self.num_neurons < 2:
            return pd.DataFrame({
                'neuron1_id': [],
                'neuron2_id': [],
                'overlap_area': [],
                'overlap_percentage': [],
                'neuron1_area': [],
                'neuron2_area': []
            })
        
        # Generate individual masks for each neuron
        neuron_masks = []
        total_areas = []
        
        for neuron in self.neurons:
            masks = neuron.generate_binary_mask()
            neuron_mask = masks['filled'] > 0
            neuron_masks.append(neuron_mask)
            total_areas.append(np.sum(neuron_mask))
        
        # Calculate pairwise overlaps
        overlap_data = []
        for i in range(len(self.neurons)):
            for j in range(i + 1, len(self.neurons)):  # Only calculate unique pairs
                overlap_mask = np.logical_and(neuron_masks[i], neuron_masks[j])
                overlap_area = np.sum(overlap_mask)
                
                # Calculate overlap percentage relative to smaller neuron
                min_neuron_area = min(total_areas[i], total_areas[j])
                overlap_percentage = (overlap_area / min_neuron_area * 100) if min_neuron_area > 0 else 0
                
                overlap_data.append({
                    'neuron1_id': self.neurons[i].neuron_id,
                    'neuron2_id': self.neurons[j].neuron_id,
                    'overlap_area': overlap_area,
                    'overlap_percentage': overlap_percentage,
                    'neuron1_area': total_areas[i],
                    'neuron2_area': total_areas[j]
                })
        
        return pd.DataFrame(overlap_data)

    def scramble_network(self, min_overlap_percentage=0, max_overlap_percentage=100, max_attempts=100):
        """
        Repeatedly randomizes neuron positions by shifting their masks until 
        overlap constraints are met.
        
        Args:
            max_attempts (int): Maximum number of attempts to meet constraints
            
        Returns:
            bool: True if constraints were met, False if max_attempts reached
        """
        if self.num_neurons < 2:
            return True
        
        original_positions = [(n.position[0], n.position[1]) for n in self.neurons]
        
        for attempt in range(max_attempts):
            # Generate new random positions within valid bounds
            for i in range(self.num_neurons):
                # Generate random position considering edge margin
                new_x = random.randint(self.edge_margin, self.width - self.edge_margin)
                new_y = random.randint(self.edge_margin, self.height - self.edge_margin)
                
                # Calculate shift needed
                dx = int(new_x - self.neurons[i].position[0])
                dy = int(new_y - self.neurons[i].position[1])
                
                # Shift the neuron's masks and update its position
                self.translate_neuron(i, dx, dy, rotation=np.random.randint(0, 360))
            
            # Calculate overlap metrics
            overlap_metrics_dict = self.calculate_overlap_metrics()
            filled_overlap_percentage = overlap_metrics_dict['filled_overlap_percentage']
            
            if (filled_overlap_percentage >= min_overlap_percentage) and (filled_overlap_percentage <= max_overlap_percentage):
                return overlap_metrics_dict
        
        # If we couldn't meet constraints, restore original positions
        for i, (orig_x, orig_y) in enumerate(original_positions):
            dx = int(orig_x - self.neurons[i].position[0])
            dy = int(orig_y - self.neurons[i].position[1])
            self.translate_neuron(i, dx, dy)
        
        return False

    def add_neuron(self, neuron=None, copy_index=None, position=None):
        """
        Add a neuron to the network, either by:
        1. Adding an existing neuron object (optionally at a new position)
        2. Copying an existing neuron from the network (optionally to a new position)
        3. Creating a new neuron at the specified position
        
        Args:
            neuron (Neuron, optional): Existing neuron object to add directly to the network
            copy_index (int, optional): Index of existing neuron to copy
            position (tuple, optional): (x, y, angle) position for new neuron or copied neuron
            
        Returns:
            int: Index of the added neuron
        """
        # Check arguments
        if neuron is not None and copy_index is not None:
            raise ValueError("Cannot specify both neuron and copy_index")
        if neuron is None and copy_index is None and position is None:
            raise ValueError("Must specify either neuron, copy_index, position, or a combination")
        
        if neuron is not None:
            # Add existing neuron object directly
            neuron.network = self
            neuron.neuron_id = f"{self.network_id}_neuron_{len(self.neurons) + 1}"
            self.neurons.append(neuron)
            self.neuron_masks.append(neuron.generate_binary_mask())
            
            # If position is provided, translate the neuron
            if position is not None:
                dx = int(position[0] - neuron.position[0])
                dy = int(position[1] - neuron.position[1])
                self.translate_neuron(len(self.neurons) - 1, dx, dy)
                if len(position) > 2:
                    neuron.starting_angle = position[2]
            
        elif copy_index is not None:
            if copy_index >= len(self.neurons):
                raise ValueError(f"copy_index {copy_index} out of range for {len(self.neurons)} neurons")
            # Copy existing neuron
            source_neuron = self.neurons[copy_index]
            new_position = position if position is not None else source_neuron.position
            
            neuron = Neuron(
                position=source_neuron.position,  # Start at source position
                starting_angle=source_neuron.starting_angle,
                **self.neuron_params,
                network=self,
                neuron_id=f"{self.network_id}_neuron_{len(self.neurons) + 1}"
            )
            self.neurons.append(neuron)
            self.neuron_masks.append(self.neuron_masks[copy_index].copy())
            
            # If position is different, translate the neuron
            if position is not None:
                dx = int(new_position[0] - source_neuron.position[0])
                dy = int(new_position[1] - source_neuron.position[1])
                self.translate_neuron(len(self.neurons) - 1, dx, dy)
                if len(new_position) > 2:
                    neuron.starting_angle = new_position[2]
            
        else:  # position only
            # Create new neuron at specified position
            neuron = Neuron(
                position=position[:2],  # x, y coordinates
                starting_angle=position[2] if len(position) > 2 else 0,  # angle if provided
                **self.neuron_params,
                network=self,
                neuron_id=f"{self.network_id}_neuron_{len(self.neurons) + 1}"
            )
            self.neurons.append(neuron)
            
            # Generate start points and grow the neuron
            neuron.generate_start_points()
            self.grow_single_neuron(len(self.neurons) - 1)
            
            # Store its masks
            self.neuron_masks.append(neuron.generate_binary_mask())
        
        self.num_neurons += 1
        self.update_network_masks()
        
        return len(self.neurons) - 1



