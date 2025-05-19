import yaml
import os

class Parameters:
    def __init__(self, yaml_file='/parameters.yaml'):
        yaml_file = os.path.dirname(os.path.abspath(__file__)) + yaml_file
        with open(yaml_file) as file:
            config = yaml.load(file, Loader=yaml.FullLoader)
            self.directory = config.get('directory')
            self.weights_directory = config.get('weights_directory')
            self.qsize = config.get('qsize')
            self.map_directory = config.get('map_directory')
            self.trajectory_directory = config.get('trajectory_directory')
            self.pcds_directory = config.get('pcds_directory')
            self.results_directory = config.get('results_directory')
            self.dataset = config.get('dataset')
            self.trajectory_name = config.get('trajectory_name')
            self.sigma_dist = config.get('MonteCarlo').get('sigma_dist')
            self.sigma_desc = config.get('MonteCarlo').get('sigma_desc')
            self.b_closest = config.get('MonteCarlo').get('b_closest')
            self.num_part = config.get('MonteCarlo').get('num_part')
            self.intermediate_layer = config.get('MonteCarlo').get('intermediate_layer')
            self.voxel_size = config.get('MonteCarlo').get('voxel_size')

PARAMETERS = Parameters()
