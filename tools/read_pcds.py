import numpy as np
from config import PARAMETERS
import open3d as o3d


class PCDReader:
    def __init__(self, trajectory_name, timestamp, normalize=False, voxel_size=0.006, threshold=0.001, files_dir = PARAMETERS.pcds_directory,dataset=PARAMETERS.dataset):
        self.dataset = dataset
        self.trajectory_name = trajectory_name
        self.timestamp = timestamp
        self.files_dir = files_dir
        self.normalize = normalize
        self.voxel_size = voxel_size
        self.threshold = threshold
        self.pcd = self.read_pcd()
        self.pcd_filtered = self.filter_pcd()

    def read_pcd(self):
        if self.dataset == "ARVC":
            directory = self.files_dir + self.trajectory_name + '/robot0/lidar/data/' + self.timestamp + '.pcd'
            pcd = o3d.io.read_point_cloud(directory)

        elif self.dataset == "NCLT":
            directory = self.files_dir + self.trajectory_name + '/' +  self.trajectory_name + '_vel/' + self.trajectory_name + '/velodyne_sync/' +self.timestamp + '.bin'
            dtype = np.dtype([('x', '<H'), ('y', '<H'), ('z', '<H'), ('intensity', 'B'), ('label', 'B')])
            data = np.fromfile(directory, dtype=dtype)
            points = np.stack([data['x'] * 0.005 - 100, data['y'] * 0.005 - 100, data['z'] * 0.005 - 100], axis=-1)
            pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))

        return pcd

    def filter_pcd(self):
        points = np.array(self.pcd.points)
        # remove outliers
        distances = np.linalg.norm(points, axis=1)
        points = points[distances < 50]

        # normalize
        if self.normalize:
            centroid = np.mean(points, axis=0)
            points -= centroid
            scale = np.max(np.linalg.norm(points, axis=1))
            points /= scale

        # downsampling
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
        pcd = pcd.voxel_down_sample(voxel_size=self.voxel_size)

        # remove ground plane
        plane_model, inliers = pcd.segment_plane(distance_threshold=self.threshold, ransac_n=3,
                                                 num_iterations=1000)
        outlier_cloud = pcd.select_by_index(inliers, invert=True)
        pcd = outlier_cloud

        return pcd
