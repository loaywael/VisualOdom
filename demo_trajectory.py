from odom import SiftOdom, OrbOdom
import numpy as np
import argparse
import cv2


parser = argparse.ArgumentParser("demo_trajectory")
parser.add_argument('--data', default='kitti', dest='data')
parser.add_argument('--model', default='orb', dest='model')
parser.add_argument('--camera', default='mono', dest='camera')
parser.add_argument('--method', default='direct', dest='matcher')


arl_src = "/home/loay/Desktop/asu_arl.mp4"
arl_K = np.float32([[697.71, 0., 649.64], [0., 697.77, 325.953], [0., 0., 1.]])
arl_D = np.float32([[-0.172,  0.026, -0.   ,  0.   , -0.   ]])

kitti_src = "/home/loay/Desktop/kitti_26.mp4"
kitti_D = np.float32([-0.3691481, 0.1968681, 0.001353473, 0.0005677587, -0.06770705])     # kitti
kitti_K = np.float32([[959.791, 0.0, 696.0217], [0.0, 956.9251, 224.1806], [0.0, 0.0, 1.0]])    # kitti


args = parser.parse_args()
video_src = kitti_src if args.data == "kitti" else arl_src
K = kitti_K if args.data == "kitti" else arl_K
D = kitti_D if args.data == "kitti" else arl_D
model = OrbOdom(K) if args.model == 'orb' else SiftOdom(K)
model.N_FEATURES = 500
def main(args):
    cap = cv2.VideoCapture(video_src)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    try:
        for i in range(frames):
            retval, frame = cap.read()
            if retval:
                try:
                    frame = cv2.undistort(frame, K, D)
                    model.track_motion(frame)
                except Exception as e:
                    print(e)
            cv2.imshow(f"{args.data}-{args.camera}-{args.model}", frame)
            cv2.waitKey(1)
    except KeyboardInterrupt: pass
    np.savez(
        f"./output/{args.data}-{args.model}-{args.camera}-visual_odom.npz", 
        trajectory=model.trajectory, 
        motion=model.camera_motion
    )
    cv2.destroyAllWindows()


if __name__ == '__main__':
    print("*"*50)
    print(f"data: {args.data},\t model: {args.model},\t camera-type: {args.camera}")
    print("*"*50)
    main(args)
    
