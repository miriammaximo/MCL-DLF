from scipy.spatial import cKDTree
import re
import os
import open3d as o3d
from tools.read_features import calculate_global_descriptor, calculate_local_descriptors
import csv
from tools.read_pcds import PCDReader
from tools.tools import *
from skimage.feature import match_descriptors
import time
from config.config import PARAMETERS
import pandas as pd

def string2array(desc):
    data = desc.strip("[]")
    numbers = re.split(r'\s+', data)
    numbers = [num for num in numbers if num.strip()]
    array = np.array(numbers, dtype=float)
    return array

class MonteCarlo:
    def __init__(self, robot_trajectory = PARAMETERS.trajectory_directory,
                map_data = PARAMETERS.map_directory,
                pcds_traj_dir = PARAMETERS.pcds_directory,
                results_directory = PARAMETERS.results_directory,
                trajectory_name = PARAMETERS.trajectory_name,
                weights_directory = PARAMETERS.weights_directory,
                dataset = PARAMETERS.dataset,
                layer_lf = PARAMETERS.intermediate_layer,
                sigma_dist = PARAMETERS.sigma_dist,
                sigma_desc = PARAMETERS.sigma_dist,
                b_closest = PARAMETERS.b_closest,
                num_part = PARAMETERS.num_part,
                voxel_size = PARAMETERS.voxel_size,
                ind_0 = 0):

        self.robot_trajectory = pd.read_csv(robot_trajectory).to_dict('records')
        self.map_data = pd.read_csv(map_data).to_dict('records')
        self.map_positions = np.array([np.array([dic["x"], dic["y"]]) for dic in self.map_data])
        self.sigma_dist = sigma_dist
        self.sigma_desc = sigma_desc
        self.B = b_closest
        self.num_part = num_part
        self.weights = np.ones(len(self.map_positions) * self.num_part)
        self.map_descriptors = [string2array(str(dic["descriptor"])) for dic in self.map_data]
        self.neff_lim = len(self.map_positions) * self.num_part * 0.8
        self.pcds_traj_dir = pcds_traj_dir
        self.trajectory_name = trajectory_name
        self.dataset = dataset
        self.layer_lf = layer_lf
        self.particles = []
        self.mcl_path = []
        self.start_msg = 0
        self.end_msg = 0
        self.ind_initial = ind_0
        self.voxel_size = voxel_size

        self.directory_results = results_directory + self.dataset + '/' + self.trajectory_name + '_' + layer_lf + '.csv'
        os.makedirs(results_directory + self.dataset, exist_ok=True)
        self.weights_directory = weights_directory

        columns = ["GroundTruth_x", "GroundTruth_y", "GroundTruth_theta", "MonteCarlo_x",
                    "MonteCarlo_y", "MonteCarlo_theta", "ICP_x", "ICP_y", "ICP_theta",
                    "Localf_X", "Localf_Y", "Localf_theta",
                    "Time_ICP", "Time_LF"]

        with open(self.directory_results, mode='w', newline='') as archivo_csv:
            write_csv = csv.writer(archivo_csv, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            write_csv.writerow(columns)

    def motion_mc(self):

        particles = []
        pose_i0 = self.start_msg
        pose_i1 = self.end_msg
        odo_i0 = compute_homogeneous_transforms(pose_i0["x_odom"], pose_i0["y_odom"],pose_i0['z_odom'],
                                                  pose_i0['alpha_odom'], pose_i0['beta_odom'], pose_i0['gamma_odom'])
        odo_i1 = compute_homogeneous_transforms(pose_i1["x_odom"], pose_i1["y_odom"], pose_i1['z_odom'],
                                                    pose_i1['alpha_odom'], pose_i1['beta_odom'], pose_i1['gamma_odom'])
        odo_rel = np.matmul(np.linalg.inv(odo_i0), odo_i1)
        dx3, dy3, z3, euler3 = transform_to_pose(odo_rel)
        heading = euler3[2]
        dx = pose_i1["x_odom"] - pose_i0["x_odom"]
        dy = pose_i1["y_odom"] - pose_i0["y_odom"]
        distance = np.sqrt(dx*dx+dy*dy)
        kdtree = cKDTree(self.map_positions)

        for i, tup in enumerate(self.particles):
            node = int(tup[3])
            x_part = tup[0] + distance * np.cos(tup[2] + heading)
            y_part = tup[1] + distance * np.sin(tup[2] + heading)
            ang_part = tup[2] + heading
            pos_particle = np.array([x_part, y_part])
            dist, nodes = kdtree.query(pos_particle, k=2)

            if nodes[0] != node:
                node2 = nodes[0]
            else:
                node2 = nodes[1]

            pos_node_2 = self.map_positions[node2]
            part = (pos_node_2[0], pos_node_2[1], ang_part, node2)
            particles.append(part)

        self.particles = particles

    def calculate_descriptor(self, timestamp):
        pcd = PCDReader(self.trajectory_name, timestamp =str(timestamp), files_dir =self.pcds_traj_dir, normalize = True, dataset = self.dataset).pcd_filtered
        descriptor = calculate_global_descriptor(pcd, weights=self.weights_directory)
        return descriptor

    def observe_mc(self):

        weights = []
        desc_i = self.calculate_descriptor(self.end_msg['timestamp'])
        kdtree = cKDTree(self.map_descriptors)
        distances, indices = kdtree.query(desc_i, k=self.B)
        closest_map_descriptors = np.array(self.map_descriptors)[indices]
        closest_map_positions = self.map_positions[indices]

        for i, tup in enumerate(self.particles):
            pos_particle = np.array(self.map_positions[tup[3]])
            desc_particle =  np.array(self.map_descriptors[tup[3]])
            wti = self.beam(closest_map_positions, pos_particle, closest_map_descriptors, desc_particle)
            wt = self.weights[i] * wti
            weights.append(wt)
            
        self.weights=weights/np.sum(weights)

    def beam(self, pos_map, pos_particle, descriptors_map, desc_particle):
        wi = 0
        mat_sigma_l = (1 / self.sigma_dist) * np.eye(2)
        mat_sigma_m = (1 / self.sigma_desc)
        distances = [(np.linalg.norm(desc_particle -descriptorsi)) for descriptorsi in descriptors_map]
        hj = distances /np.sum(distances)
        vj = [(pos_particle - posi) for posi in pos_map]
        for i in range(len(pos_map)):
            wi += np.exp(-np.dot(vj[i], np.dot(mat_sigma_l, vj[i].T)))* np.exp(-hj[i ] *hj[i ] *mat_sigma_m)
        return wi

    def calculate_neff(self):
        neff = 1/ np.sum([i ** 2 for i in self.weights])
        return neff

    def resample(self, weights):
        particles = []
        w = []
        norm_weights = weights / np.sum(weights)
        indexes = [np.random.choice(np.arange(0, len(self.particles)),
                                    p=norm_weights) for i in range(len(self.particles))]
        for i in indexes:
            particles.append(self.particles[i])
            w.append(self.weights[i])
        self.weights = w
        self.particles = particles


    def calculate_particles_position(self):
        positions = []
        for j, tup in enumerate(self.particles):
            pos_particle = np.array((self.map_positions[tup[3]]))
            positions.append(pos_particle)
        return np.array(positions)

    def initialization(self):
        particles = []
        for i, pos in enumerate(self.map_positions):
            pos_node = self.map_positions[i]
            rand = np.random.rand
            for j in range(self.num_part):
                rot = -np.pi + 2 * np.pi * rand(1)[0]
                transf_particle = (pos_node[0], pos_node[1], rot, i)
                particle = transf_particle
                particles.append(particle)

        self.particles = particles

    def calculate_mcl_pose(self):
        particles_positions = np.array(self.particles)[:, :2]
        if len(self.mcl_path) == 0:
            ang_mcl = 0

        else:
            pos_i_0 = np.array(self.mcl_path)[-1, 0:2]
            pos_i = np.array([np.mean(particles_positions[:, 0]),
                              np.mean(particles_positions[:, 1])])
            d_pos = pos_i - pos_i_0
            ang_mcl = np.arctan2(d_pos[1], d_pos[0])

        mcl_i = np.array([np.mean(particles_positions[:, 0]),
                          np.mean(particles_positions[:, 1]), ang_mcl])
        return mcl_i

    def save_data(self, mcl_pose, icp_pose, loc_feat, icp_time, loc_feat_time):
        gtruth_i = np.array([self.end_msg["x"], self.end_msg["y"], self.end_msg["gamma"]])
        print(f"Ground truth: {gtruth_i} - MCL: {mcl_pose} - ICP: {icp_pose} - LF: {loc_feat}")

        with open(self.directory_results, mode='a', newline='') as archivo_csv:
            write_csv = csv.writer(archivo_csv, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            matrix = [gtruth_i[0], gtruth_i[1], gtruth_i[2],
                      mcl_pose[0], mcl_pose[1], mcl_pose[2],
                      icp_pose[0], icp_pose[1], icp_pose[2],
                      loc_feat[0], loc_feat[1], loc_feat[2],
                      icp_time, loc_feat_time]
            write_csv.writerow(matrix)

    def localize_mc(self):
        self.initialization()
        self.start_msg = self.robot_trajectory[self.ind_initial]

        distance_step = 0
        steps = 0
        for i, message in enumerate(self.robot_trajectory):
            if i <=self.ind_initial: continue

            self.end_msg = message

            end_odom = (self.end_msg["x_odom"], self.end_msg["y_odom"])

            distance_step = distance_step + (np.linalg.norm(np.array(end_odom) -np.array(
                [self.robot_trajectory[i - 1]["x_odom"], self.robot_trajectory[i - 1]["y_odom"]])))

            if distance_step < 1: continue
            distance_step = 0
            print('Moving particles...')
            self.motion_mc()
            print('Weighting particles...')
            self.observe_mc()
            neff_i = self.calculate_neff()
            if neff_i < self.neff_lim:
                print('Resampling particles...')
                self.resample(self.weights)
                self.weights = np.ones(len(self.particles))

            self.start_msg = self.end_msg
            mcl_pose = self.calculate_mcl_pose()
            self.mcl_path.append(mcl_pose)
            print('Global registration with ICP...')
            icp_pose, icp_time = self.refine_pose_with_icp(mcl_pose)
            print('Global registration with DLF...')
            loc_feat, loc_feat_time =  self.refine_pose_with_local_desc(mcl_pose)

            self.save_data(mcl_pose, icp_pose, loc_feat, icp_time, loc_feat_time)

            steps += 1
            if steps>20:
                self.initialization()
                self.weights = np.ones(len(self.particles))
                steps = 0

    def refine_pose_with_icp(self, mcl_pose):
        start_time = time.time()
        x_mcl, y_mcl, gamma_mcl = mcl_pose[0], mcl_pose[1], mcl_pose[2]

        kdtree = cKDTree(self.map_positions)
        dist, node_map = kdtree.query(mcl_pose[0:2], k=1)

        ruta = str(self.map_data[node_map]['id_map'])
        timestamp = str(self.map_data[node_map]['timestamp'])

        pcd_map = PCDReader(ruta, timestamp, dataset=self.dataset,files_dir =self.pcds_traj_dir).pcd_filtered

        ruta = str(self.trajectory_name)
        timestamp = str(self.end_msg['timestamp'])
        pcd_robot = PCDReader(ruta, timestamp, dataset=self.dataset, files_dir =self.pcds_traj_dir).pcd_filtered

        voxel_size_normals = 0.5
        max_nn_gd = 300
        threshold = 1

        data_map = self.map_data[node_map]
        x_map, y_map = data_map['x'], data_map['y']
        gamma_map = data_map['gamma']

        T_pcd_robot = compute_homogeneous_transforms(x_mcl, y_mcl, 0, 0, 0, gamma_mcl)

        T_pcd_map = compute_homogeneous_transforms(x_map, y_map, 0, 0, 0, gamma_map)

        transformation_initial = np.matmul(np.linalg.inv(T_pcd_map), T_pcd_robot)

        pcd_map.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size_normals, max_nn=max_nn_gd))
        pcd_robot.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size_normals, max_nn=max_nn_gd))

        corrected_transform = o3d.pipelines.registration.registration_icp(
            pcd_robot, pcd_map, threshold, transformation_initial,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000)
        )

        transformation_global = np.matmul(T_pcd_map, np.array(corrected_transform.transformation))

        x_icp, y_icp, z_icp, euler_icp = transform_to_pose(np.array(transformation_global))

        icp_time = time.time() - start_time
        return np.array([x_icp, y_icp, euler_icp[2]]), icp_time

    def refine_pose_with_local_desc(self, mcl_pose):
        start_time = time.time()
        kdtree = cKDTree(self.map_positions)
        dist, node_map = kdtree.query(mcl_pose[0:2], k=1)

        ruta = str(self.map_data[node_map]['id_map'])
        timestamp = str(self.map_data[node_map]['timestamp'])
        pcd_map = PCDReader(ruta, timestamp, normalize=False, voxel_size = self.voxel_size, threshold =0.1, dataset =self.dataset, files_dir =self.pcds_traj_dir).pcd_filtered #voxel_size (NLCT=0.1) (ARVC =0.25)

        coordinates_map, local_descriptors_map = calculate_local_descriptors(pcd_map, self.layer_lf)

        ruta = str(self.trajectory_name)
        timestamp = str(self.end_msg['timestamp'])
        pcd_robot = PCDReader(ruta, timestamp, normalize=False, voxel_size = self.voxel_size, threshold =0.1, dataset =self.dataset,files_dir =self.pcds_traj_dir).pcd_filtered  #voxel_size (NLCT=0.1)(ARVC =0.25)

        coordinates_robot, local_descriptors_robot = calculate_local_descriptors(pcd_robot, self.layer_lf)

        matches = match_descriptors(local_descriptors_robot, local_descriptors_map, cross_check=True, metric='euclidean')

        voxel_size_normals = 2
        max_nn_gd = 30
        data_map = self.map_data[node_map]
        x_map, y_map = data_map['x'], data_map['y']
        gamma_map = data_map['gamma']
        T_pcd_map = compute_homogeneous_transforms(x_map, y_map, 0, 0, 0, gamma_map)

        pcd_map_coord = o3d.geometry.PointCloud()
        pcd_map_coord.points = o3d.utility.Vector3dVector(coordinates_map[:, 1:]*0.01)
        pcd_map_coord.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size_normals, max_nn=max_nn_gd))

        pcd_robot_coord = o3d.geometry.PointCloud()
        pcd_robot_coord.points = o3d.utility.Vector3dVector(coordinates_robot[:, 1:] * 0.01)
        pcd_robot_coord.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size_normals, max_nn=max_nn_gd))

        corr = o3d.utility.Vector2iVector(matches)
        result = o3d.pipelines.registration.registration_ransac_based_on_correspondence(pcd_robot_coord,  pcd_map_coord,
                                                                                        corr,1)
        
        transformation_global = np.matmul(T_pcd_map, np.array(result.transformation))

        x_lf, y_lf, z_lf, euler_lf = transform_to_pose(np.array(transformation_global))
        lf_time = time.time() - start_time
        return np.array([x_lf, y_lf, euler_lf[2]]), lf_time