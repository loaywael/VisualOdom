#!/usr/bin/env python3
from camera import Camera
import open3d as o3d
import numpy as np
import threading
import pykitti
import time


class Scenes3DVisualizer:
    def __init__(self, win_title="Open3D", win_size=(1920, 1080),
        show_axes=True, show_ground=True) -> None:
        '''Scene Constructor'''
        self._app = o3d.visualization.gui.Application.instance
        self._app.initialize()
        self._win = o3d.visualization.O3DVisualizer(
            title=win_title, width=win_size[0], height=win_size[1],
        )
        self._win.show_skybox(False)
        self._win.show_menu(False)
        self._win.set_background(np.float32([0]*4), None)
        self._win.show_ground = show_ground
        self._app.add_window(self._win)
        self._is_view_initialized = False

    def add_geometries(self, geometries):
        '''Updates the scene'''
        for geometry in geometries:
            self._win.remove_geometry(geometry["name"])
            self._win.add_geometry(geometry)
        
    def show(self, line_width=3, point_size=3, 
        save=False, filename=None, interactive=False):
        '''View the drawn scene'''
        # self._win.show(True)
        self._win.post_redraw()
        self._win.line_width = line_width
        self._win.point_size = point_size
        if save: 
            if filename is None: file = "/output/tmp.png"
            self._win.capture_screen_image(filename)
        if interactive: self._app.run()
        else: self._app.run_one_tick()
        if not self._is_view_initialized: 
            self._win.reset_camera_to_default()
            self._is_view_initialized = True

    def close(self):
        self._win.close()
        self._app.quit()

    # >>>>>>>> proposed update_view_method
    # bounds = self._win.scene.bounding_box
    # self._win.setup_camera(60.0, bounds, bounds.get_center())
    # self._app.post_to_main_thread(self._win, self._draw_geometries_thread)


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
    pcd_obj : (PointCloud)
        open3d.geometry.PointCloud object 
    pcd_arr : (np.float)
        LinSet Mesh (points, lines) f the camera view 5 points
    color : (list)
    tf : (list)
        object transformation matrix
    '''
    pcd_obj.point["positions"] = np.float32(pcd_arr)
    if color is not None:
        pcd_obj.point["colors"] = np.float32([color]*len(pcd_arr))
    if tf is not None:
        pcd_obj.transform(tf)


def set_camera_view(lineset_obj, pose, color=[0, 1, 0], tf=None):
    '''
    Params
    ------
    lineset_obj : (LineSet)
        open3d.geometry.LineSet object
    pose : (dict)
        LinSet Mesh (points, lines) f the camera view 5 points
    color : (list)
    tf : (list)
        object transformation matrix
    '''
    lineset_obj.point["positions"] = np.float32(pose['points'])
    lineset_obj.line["indices"] = np.float32(pose['lines'])
    lineset_obj.line["colors"] = np.float32([color]*len(pose['lines']))
    if tf is not None:
        lineset_obj.transform(tf)


if __name__ == '__main__':
    # o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)
    img_to_pcd_tf = [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
    kitti_src = "/home/loay/Documents/datasets/kitti/raw/odometry_gray/dataset"
    kitti_odom = pykitti.odometry(kitti_src, "03")
    pred_path = np.load('output/kitti-orb-1000-mono-visual_odom.npz')['trajectory']
    gt_path = np.array([pose[:3, 3] for pose in kitti_odom.poses])
    pcdObj1 = o3d.t.geometry.PointCloud()
    set_pointcloud_obj(pcdObj1, gt_path, color=[0, 0, 1], tf=img_to_pcd_tf)
    
    pcdObj2 = o3d.t.geometry.PointCloud()
    vis = Scenes3DVisualizer("Odom", (0, 960), (960, 1080))

    try:
        pred_traj = []
        vis.add_geometries([{"name":"gt_path", "geometry": pcdObj1}])
        for i in range(len(pred_path)):
            pred_traj.extend(pred_path[i:i+50])
            set_pointcloud_obj(pcdObj2, pred_traj, color=[1, 0, 0], tf=img_to_pcd_tf)
            vis.add_geometries([{"name":"pred_path", "geometry": pcdObj2}])
            vis.show(interactive=False)
            # time.sleep(0.01)
        vis.close()
    except Exception as e: print(e)  
