#!/usr/bin/env python3
from visualization import Scenes3DVisualizer
from visualization import set_pointcloud_obj
from visualization import set_camera_view
from odom import SiftOdom, OrbOdom
from camera import Camera
import open3d as o3d
import numpy as np
import argparse
import pykitti
import cv2


parser = argparse.ArgumentParser("demo_trajectory")
parser.add_argument('--data', default='kitti', dest='data')
parser.add_argument('--sequence', type=int, default=7, dest='vid_seq')
parser.add_argument('--model', default='orb', dest='model')
parser.add_argument('--nbr-features', type=int, default=500, dest='nbr_features')
parser.add_argument('--camera', default='mono', dest='camera')
parser.add_argument('--method', default='direct', dest='matcher')
parser.add_argument('--save', default=False, dest='save')
args = parser.parse_args()

cv_win_title = f"{args.data}-{args.camera}-{args.model}"
kitti_src = "/home/loay/Documents/datasets/kitti/raw/odometry_gray/dataset"
kitti_odom = pykitti.odometry(kitti_src, "%02d"%args.vid_seq)

cam0 = Camera(*(960, 290), kitti_odom.calib.P_rect_00)
odom_params = (cam0.K, args.nbr_features)
model = OrbOdom(*odom_params) if args.model == 'orb' else SiftOdom(*odom_params)
img_to_pcd_tf = [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
gt_path = np.array([pose[:3, 3] for pose in kitti_odom.poses])

pcd_gt_path = o3d.t.geometry.PointCloud()
pcd_pred_path = o3d.t.geometry.PointCloud()
cam_view_mesh = o3d.t.geometry.LineSet()

set_pointcloud_obj(pcd_gt_path, np.array(gt_path), color=[0, 0, 1], tf=img_to_pcd_tf)
vis3d = Scenes3DVisualizer("Odom")
vis3d.add_geometries([{"name":"gt_path", "geometry": pcd_gt_path}])
cv2.namedWindow(cv_win_title)


def main(args):
    try:
        for frame_id , img in enumerate(kitti_odom.cam0):
            frame = np.asarray(img)
            try:
                model.track_motion(frame)
                frame = cv2.resize(frame, (960, 290))
                set_pointcloud_obj(pcd_pred_path, model.trajectory, color=[1, 0, 0], tf=img_to_pcd_tf)
                cam0.setProjectionMtx(model.cam_tf[-1][:3])         # last TF -> P(3x4)
                set_camera_view(cam_view_mesh, cam0.view(20), tf=img_to_pcd_tf)
                vis3d.add_geometries([
                    {"name":"pred_path", "geometry": pcd_pred_path},
                    {"name":"cam_view", "geometry": cam_view_mesh},
                ])
                vis3d.show(interactive=False, line_width=3, point_size=3)
            except Exception as e: print(e)
            cv2.imshow(cv_win_title, frame)
            cv2.moveWindow(cv_win_title, 0, 0)
            if cv2.waitKey(1) == ord('q'): break
    except KeyboardInterrupt: pass
    if args.save:
        np.savez(
            f"./output/{args.data}-{args.model}-{args.nbr_features}-{args.camera}-visual_odom.npz", 
            trajectory=model.trajectory, 
            motion=model.cam_tf
        )
    cv2.destroyAllWindows()


if __name__ == '__main__':
    print("*"*50)
    print(f"data: {args.data},\t model: {args.model},\t camera-type: {args.camera}")
    print("*"*50)
    main(args)
    
