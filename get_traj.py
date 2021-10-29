from odom import SiftOdom, OrbOdom
import numpy as np
import cv2


video_src = "/home/loay/Desktop/test_scene.mp4"
cap = cv2.VideoCapture(video_src)
frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# calib_data = np.load("calib/calib_data.npz", allow_pickle=True)
K = np.float32([[697.71, 0., 649.64], [0., 697.77, 325.953], [0., 0., 1.]])
D = np.float32([[-0.172,  0.026, -0.   ,  0.   , -0.   ]])
# model = SiftOdom(K)
model = OrbOdom(K)

try:
    for i in list(range(frames))[:-1]:
        retval, frame = cap.read()
        if retval:
            try:
                point3d = model(frame)
            except Exception as e:
                print(e)
        cv2.imshow("", frame)
        delay = int((1/fps)*1000)
        cv2.waitKey(1)

except KeyboardInterrupt: pass

np.savez("points3d.npz", points3d=model.trajectory)
cv2.destroyAllWindows()
