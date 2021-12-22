from abc import abstractmethod, ABC
from collections import deque
import numpy as np
import time
import cv2


def timit(func):
    def wrapper(*args, **kwargs):
        tick = time.perf_counter()
        res = func(*args, **kwargs)
        tock = time.perf_counter()
        print(f"[runtime] --- {func.__name__}:  {(tock-tick):.3f}(s)")
        return res
    return wrapper


class VisualOdom(ABC):
    @abstractmethod
    def estimate_motion(self):
        pass

    @abstractmethod
    def track_motion(self):
        pass


class MonoCamVisualOdom(VisualOdom):
    def __init__(self, intrinsic_mtx):
        self._K = intrinsic_mtx
        self._kpts_buffer = deque(maxlen=2)
        self._desc_buffer = deque(maxlen=2)
        self._frames_buffer = deque(maxlen=2)
        self._camTFs = [np.eye(4)]
    
    # @timit
    def estimate_motion(self, img1Pts, img2Pts, K):
        E, mask = cv2.findEssentialMat(img1Pts, img2Pts, K, method=cv2.RANSAC, prob=0.999, threshold=0.75)
        ret, camR, camT, mask = cv2.recoverPose(E, img1Pts, img2Pts, K)
        camTF = np.eye(4)   # (TF) -> world origin with respect to the camera center
        camTF[:-1, :-1] = camR.T
        camTF[:-1, -1:] = -camR.T @ camT
        self._camTFs.append(np.mean(self._camTFs[-1:-2:-1], axis=0) @ camTF)    # smooth by the last 3 TFs

    @abstractmethod
    def get_features(self, frame:np.ndarray)-> tuple:
        pass

    @abstractmethod
    def match_features(self, desc1:list, desc2:list): 
        pass

    @staticmethod
    def get_matches(matches:list, img1Kpts:list, img2Kpts:list):
        img1_pts = []
        img2_pts = []
        for match in matches:
            img1_pts.append(img1Kpts[match.queryIdx].pt)
            img2_pts.append(img2Kpts[match.trainIdx].pt)
        return np.array(img1_pts), np.array(img2_pts)

    @property
    def trajectory(self):
        return np.array(self._camTFs)[:, :3, 3]

    @property
    def camera_motion(self):
        return np.array(self._camTFs)

    @timit
    def track_motion(self, frame):
        self._frames_buffer.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        kpts, des = self.get_features(frame)
        self._kpts_buffer.append(kpts)
        self._desc_buffer.append(des)
        if len(self._frames_buffer) == self._frames_buffer.maxlen:
            matched_pts = self.match_features(*self._desc_buffer)
            img1_pts, img2_pts = MonoCamVisualOdom.get_matches(matched_pts, *self._kpts_buffer)
            self.estimate_motion(img1_pts, img2_pts, self._K)


class SiftOdom(MonoCamVisualOdom):
    SIFT_N_FEATURES = 100
    SIFT_CONTRAST_THRESHOLD = 0.15
    SIFT_EDGE_THRESHOLD = 15
    SIFT_N_OCTAVE_LAYERS = 5

    def __init__(self, *args, **kwargs):
        super(SiftOdom, self).__init__(*args, **kwargs)
        self._matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        self._sift = cv2.SIFT_create(
            nfeatures=kwargs.get('nfeatures', SiftOdom.SIFT_N_FEATURES), 
            contrastThreshold=kwargs.get('contrastThreshold', SiftOdom.SIFT_CONTRAST_THRESHOLD), 
            edgeThreshold=kwargs.get('edgeThreshold', SiftOdom.SIFT_EDGE_THRESHOLD), 
            nOctaveLayers=kwargs.get('nOctaveLayers', SiftOdom.SIFT_N_OCTAVE_LAYERS)
        )

    def get_features(self, frame:np.uint8):
        kpts, des = self._sift.detectAndCompute(frame, None)
        return kpts, des

    def match_features(self, desc1:list, desc2:list):
        matched_pts = self._matcher.match(desc1, desc2)
        sortedMatches = sorted(matched_pts, key=lambda x: x.distance)[:50]
        return sortedMatches

 
class OrbOdom(MonoCamVisualOdom):
    ORB_N_FEATURES = 500    # more features more stable odom
    ORB_EDGE_THRESHOLD = 13
    ORB_SCORE_TYPE = cv2.ORB_FAST_SCORE

    def __init__(self, *args, **kwargs):
        super(OrbOdom, self).__init__(*args, **kwargs)
        self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self._orb = cv2.ORB_create(
            nfeatures=kwargs.get('nfeatures', OrbOdom.ORB_N_FEATURES), 
            edgeThreshold=kwargs.get('edgeThreshold', OrbOdom.ORB_EDGE_THRESHOLD), 
            scoreType=kwargs.get('scoreType', OrbOdom.ORB_SCORE_TYPE),
        )

    def get_features(self, frame:np.uint8):
        kpts, des = self._orb.detectAndCompute(frame, None)
        return kpts, des

    def match_features(self, desc1:list, desc2:list):
        matched_pts = self._matcher.match(desc1, desc2)
        sortedMatches = sorted(matched_pts, key=lambda x: x.distance)
        return sortedMatches


