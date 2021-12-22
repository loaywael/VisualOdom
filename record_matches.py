#!/usr/bin/python3
import numpy as np
import time
import cv2


video_src = "/home/loay/Desktop/asu_arl.mp4"
cap = cv2.VideoCapture(video_src)
frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

K = np.float32([[697.71, 0., 649.64], [0., 697.77, 325.953], [0., 0., 1.]])
D = np.float32([[-0.172,  0.026, -0.   ,  0.   , -0.   ]])

frames_buffer = []
kpts_buffer = []
desc_buffer = []
orb = cv2.ORB_create(nfeatures=500, scoreType=cv2.ORB_HARRIS_SCORE, edgeThreshold=13)
bfMatcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True) # double match and doubls runtime
flags = cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS + cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS

for frameIdx in range(frames):
    retval, frame = cap.read()
    if retval:
        try: 
            frame = cv2.undistort(frame, K, D)
            grayImg = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frames_buffer.append(grayImg) 
            t1 = time.perf_counter()
            kpts, desc = orb.detectAndCompute(grayImg, None)
            kpts_buffer.append(kpts)
            desc_buffer.append(desc)
            # print("here", frameIdx % 2 == 0, len(frames_buffer))
            if len(frames_buffer) == 2:
                matches = bfMatcher.match(*desc_buffer)
                sortedMatches = sorted(matches, key=lambda x: x.distance)   # (1 to 1) min L2 is better
                t2 = time.perf_counter()
                print(f"BFMatcher runtime: {(t2 -t1)*1000:0.2f} (ms)")
                matchImg = cv2.drawMatches(
                    frames_buffer[0], kpts_buffer[0], 
                    frames_buffer[1], kpts_buffer[1], 
                    sortedMatches[:50], outImg=None, flags=flags
                )
                matchImg = cv2.resize(matchImg, None, None, fx=0.75, fy=0.75)
                cv2.imwrite(f"./matches/{frameIdx:06d}.png", matchImg)
                cv2.imshow("matches", matchImg)
                cv2.waitKey(1)
                frames_buffer.pop(0)
                kpts_buffer.pop(0)
                desc_buffer.pop(0)
        except Exception as e: print(e)
cv2.destroyAllWindows()
