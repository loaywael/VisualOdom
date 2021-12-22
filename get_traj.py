from odom import SiftOdom, OrbOdom
import numpy as np
import cv2


video_src = "/home/loay/Desktop/kitti_26.mp4"
video_src = "/home/loay/Desktop/asu_arl.mp4"

cap = cv2.VideoCapture(video_src)
frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

K = np.float32([[697.71, 0., 649.64], [0., 697.77, 325.953], [0., 0., 1.]])
D = np.float32([[-0.172,  0.026, -0.   ,  0.   , -0.   ]])
# D = np.float32([-0.3691481, 0.1968681, 0.001353473, 0.0005677587, -0.06770705])     # kitti
# K = np.float32([[959.791, 0.0, 696.0217], [0.0, 956.9251, 224.1806], [0.0, 0.0, 1.0]])    # kitti
# model = SiftOdom(K)
model = OrbOdom(K)
try:
    for i in range(frames):
        retval, frame = cap.read()
        if retval:
            try:
                frame = cv2.undistort(frame, K, D)
                model.track_motion(frame)
            except Exception as e:
                print(e)
        cv2.imshow("", frame)
        cv2.waitKey(1)

except KeyboardInterrupt: pass
np.savez("visual_odom.npz", trajectory=model.trajectory, motion=model.camera_motion)
cv2.destroyAllWindows()
