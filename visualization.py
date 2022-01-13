#!/usr/bin/env python3
from camera import Camera
import open3d as o3d
import numpy as np
import pykitti
import time


class Interactive3DVisualizer:
    def __init__(self, win_title, win_loc, win_size) -> None:
        self._vis = o3d.visualization.Visualizer()
        self._win = self._vis.create_window(
            window_name=win_title,
            width=win_size[0],
            height=win_size[1],
            top=win_loc[0],
            left=win_loc[1],
        )
        self._ctr = self._vis.get_view_control()

    def add_geometries(self, geometries):
        for geometry in geometries:
            self._vis.add_geometry(geometry)
    
    def show(self, geometry, save=False, filename="/output/tmp.png"):
        self._vis.update_geometry(geometry)
        self._vis.poll_events()
        self._vis.update_renderer()
        if save: self._vis.capture_screen_image(filename)


def set_pointcloud_obj(pcd_obj, pcd_arr, color=None, tf=None):
    '''
    Params
    ------
    pose : (dict)
        LinSet Mesh (points, lines) f the camera view 5 points
    '''
    pcd_obj.points = o3d.utility.Vector3dVector(pcd_arr)
    if color is not None:
        pcd_obj.paint_uniform_color(color)
    if tf is not None:
        pcd_obj.transform(tf)


def set_camera_view(lineset_obj, pose, color=[0, 1, 0], tf=None):
    '''
    Params
    ------
    pose : (dict)
        LinSet Mesh (points, lines) f the camera view 5 points
    '''
    lineset_obj.points = o3d.utility.Vector3dVector(pose['points'])
    lineset_obj.lines = o3d.utility.Vector2iVector(pose['lines'])
    lineset_obj.paint_uniform_color(color)
    if tf is not None:
        lineset_obj.transform(tf)


if __name__ == '__main__':
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)
    img_to_pcd_tf = [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
    kitti_src = "/home/loay/Documents/datasets/kitti/raw/odometry_gray/dataset"
    kitti_odom = pykitti.odometry(kitti_src, "07")
    pred_path = np.load('output/kitti-orb-mono-visual_odom.npz')['trajectory']
    gt_path = np.array([pose[:3, 3] for pose in kitti_odom.poses])
    pcdObj1 = o3d.geometry.PointCloud()
    pcdObj1.points = o3d.utility.Vector3dVector(np.array(gt_path))
    pcdObj1.paint_uniform_color([0, 0, 1])
    pcdObj1.transform(img_to_pcd_tf)
    pcdObj2 = o3d.geometry.PointCloud()
    vis = Interactive3DVisualizer("Odom", (0, 960), (960, 1080))

    try:
        pred_traj = []
        vis.add_geometries([pcdObj1, pcdObj2])
        for i in range(0, 1000):
            pred_traj.extend(pred_path[i:i+50])
            pcdObj2.points = o3d.utility.Vector3dVector(np.array(pred_traj))
            pcdObj2.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
            pcdObj2.paint_uniform_color([1, 0, 0])
            # time.sleep(0.01)
            vis.show(pcdObj2)
    except Exception as e: print(e)  
