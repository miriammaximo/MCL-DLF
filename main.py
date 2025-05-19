from tools.montecarlo import MonteCarlo
from config.config import PARAMETERS

if __name__ == '__main__':

    num_part = PARAMETERS.num_part
    sigma_dist = PARAMETERS.sigma_dist
    sigma_desc = PARAMETERS.sigma_desc
    b_closest = PARAMETERS.b_closest
    trajectory_name = PARAMETERS.trajectory_name
    pcd_dir = PARAMETERS.pcds_directory
    map_directory = PARAMETERS.map_directory
    trajectory_directory = PARAMETERS.trajectory_directory
    results_directory = PARAMETERS.results_directory
    weights = PARAMETERS.weights_directory
    dataset = PARAMETERS.dataset
    voxel_size = PARAMETERS.voxel_size

    mc = MonteCarlo(robot_trajectory=trajectory_directory, map_data=map_directory,
                    pcds_traj_dir=pcd_dir, results_directory=results_directory,
                    trajectory_name=trajectory_name, weights_directory=weights,
                    dataset=dataset, sigma_dist=sigma_dist,
                    sigma_desc=sigma_desc, b_closest=b_closest,
                    num_part=num_part, voxel_size=voxel_size)

    mc.localize_mc()
