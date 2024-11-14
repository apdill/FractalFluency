import numpy as np
import cv2
import random
from scipy.interpolate import splprep, splev

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


    # Create and generate the network
    network = Network(width=network_width, 
                      height=network_height, 
                      num_neurons=num_neurons, 
                      neuron_params=neuron_params, 
                      edge_margin=edge_margin,
                      network_id=network_id)
    

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
        theta = np.linspace(0, 2 * np.pi, 100)
        sine_variation = np.random.uniform(0, 15) * np.sin(2 * theta)
        gaussian_variation = np.random.normal(0, 2, len(theta))
        ellipse_ratio = np.random.uniform(0.8, 1.2)
        elongation_angle = np.random.uniform(0, 2 * np.pi)

        self.x_soma = (self.radius + gaussian_variation + sine_variation) * (
            np.cos(theta) * np.cos(elongation_angle)
            - np.sin(theta) * np.sin(elongation_angle) * ellipse_ratio
        ) + self.position[0]
        self.y_soma = (self.radius + gaussian_variation + sine_variation) * (
            np.sin(theta) * np.cos(elongation_angle)
            + np.cos(theta) * np.sin(elongation_angle) * ellipse_ratio
        ) + self.position[1]

    def create_binary_masks(self, size):
        """
        Creates both filled and outline binary masks of the soma using stored coordinates.

        Args:
            size (tuple): The size of the masks.

        Returns:
            dict: A dictionary containing 'filled' and 'outline' masks.
        """
        coordinates = np.array([self.x_soma, self.y_soma]).T.astype(np.int32)

        # Create the filled mask
        mask_filled = np.zeros(size, dtype=np.uint8)
        cv2.fillPoly(mask_filled, [coordinates], 1)

        # Create the outline mask
        mask_outline = np.zeros(size, dtype=np.uint8)
        cv2.polylines(mask_outline, [coordinates], isClosed=True, color=1, thickness=1)

        return {'filled': mask_filled, 'outline': mask_outline}



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
        total_length = 500,
        initial_thickness = 10
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

        num_branches = int(np.clip(np.round(np.random.normal(self.mean_branches, 1)), 1, None))
        new_branches = []

        for i in range(num_branches):
            new_angle = angle + self.branch_angle * (i - (num_branches - 1) / 2)
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
        total_length = 500
    ):
        self.network = network
        self.position = position
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
            initial_thickness
        )

        self.neuron_id = neuron_id
        self.current_depth = 0
        self.branch_ends = []
        self.is_growing = True  # Flag to indicate if the neuron is still growing

    def generate_start_points(self):
        """
        Generates the starting points for the primary dendrites and initializes branch ends.
        """
        x_soma = self.soma.x_soma
        y_soma = self.soma.y_soma

        num_soma_points = len(x_soma)
        base_indices = np.linspace(
            0, num_soma_points - 1, self.dendrite.n_primary_dendrites, endpoint=False
        ).astype(int)

        random_offsets = np.random.randint(
            -num_soma_points // (100 // self.dendrite.n_primary_dendrites // 1.5),
            (100 // self.dendrite.n_primary_dendrites // 1.5) + 1,
            size=self.dendrite.n_primary_dendrites,
        )
        random_indices = (base_indices + random_offsets) % num_soma_points

        start_points = []
        for index in random_indices:
            start_points.append((x_soma[index], y_soma[index]))

        # Initialize branch ends with angles pointing outward from the soma
        for point in start_points:
            dx = point[0] - self.position[0]
            dy = point[1] - self.position[1]
            angle = np.arctan2(dy, dx)
            self.branch_ends.append((point, angle))


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

    def add_branches(self, accepted_branches, network_mask_filled, network_mask_outline):
        """
        Add the accepted branches directly to the network masks and update branch ends.

        Args:
            accepted_branches (list): List of branches to add.
            network_mask_filled (np.ndarray): The shared filled network mask to draw on.
            network_mask_outline (np.ndarray): The shared outline network mask to draw on.
        """
        new_branch_ends = []

        for branch_info in accepted_branches:
            branch_data = branch_info['branch_data']
            points = branch_data['points']
            new_branches = branch_info['new_branches']

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
                # Draw outline
                cv2.line(
                    network_mask_outline,
                    tuple(coordinates[i]),
                    tuple(coordinates[i + 1]),
                    1,
                    thickness=1  # Fixed outline thickness
                )

            del branch_data
            new_branch_ends.extend(new_branches)

        self.branch_ends = new_branch_ends



class Network:
    """
    Represents a network of neurons.
    """

    def __init__(self, width, height, num_neurons, neuron_params, edge_margin=100, network_id = None):
        self.width = width
        self.height = height
        self.num_neurons = num_neurons
        self.neuron_params = neuron_params
        self.neurons = []
        self.network_mask = np.zeros((self.height, self.width), dtype=np.uint8)
        self.somas_mask = np.zeros((self.height, self.width), dtype=np.uint8)
        self.network_id = network_id
        self.edge_margin = edge_margin  # Fixed edge margin of 100 pixels

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
        if not hasattr(self, 'network_mask_filled'):
            self.network_mask_filled = np.zeros((self.height, self.width), dtype=np.uint8)
        if not hasattr(self, 'network_mask_outline'):
            self.network_mask_outline = np.zeros((self.height, self.width), dtype=np.uint8)

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

            self.network_mask_filled = np.logical_or(self.network_mask_filled, soma_masks['filled']).astype(np.uint8)
            self.network_mask_outline = np.logical_or(self.network_mask_outline, soma_masks['outline']).astype(np.uint8)

            # Add neuron to the network
            self.neurons.append(neuron)

            # Generate start points for dendrite growth
            neuron.generate_start_points()

    def grow_network(self):
        """
        Grows the dendrites of all neurons in the network layer by layer, drawing directly onto the network masks.
        """
        # Use the existing network masks that already have somas drawn

        growing = True
        while growing:
            growing = False
            for neuron in self.neurons:
                if neuron.is_growing:
                    proposed_branches = neuron.prepare_next_layer()
                    if proposed_branches:
                        neuron.add_branches(proposed_branches, self.network_mask_filled, self.network_mask_outline)
                        growing = True
                    else:
                        neuron.is_growing = False

            # Increment the current depth for all neurons
            for neuron in self.neurons:
                if neuron.is_growing:
                    neuron.current_depth += 1



    def generate_binary_mask(self):
        """
        Returns the network masks generated during growth.

        Returns:
            dict: A dictionary containing 'filled' and 'outline' network masks.
        """
        return {
            'filled': self.network_mask_filled,
            'outline': self.network_mask_outline
        }


