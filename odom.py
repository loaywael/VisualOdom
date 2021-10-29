from abc import abstractmethod, ABC
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
    def estimate_motion():
        pass

    @abstractmethod
    def estimate_trajectory():
        pass


class MonoCamVisualOdom(VisualOdom):
    def __init__(self, intrinsic_mtx):
        self.__K = intrinsic_mtx
        self.__kpts_buffer = []
        self.__des_buffer = []
        self.__trajectory = []
        self.__pair_buffer = []
        self.__P = np.eye(4)
        self.__P_next = self.__P.copy()
    
    @staticmethod
    # @timit
    def estimate_motion(matches, img1_kpts, img2_kpts, K):
        img1_pts = []
        img2_pts = []
        for match in matches:
            img1_pts.append(img1_kpts[match.queryIdx].pt)
            img2_pts.append(img2_kpts[match.trainIdx].pt)
        pts1, pts2 = np.array(img1_pts), np.array(img2_pts)
        E, mask = cv2.findEssentialMat(pts1, pts2, K)
        ret, R, t, mask = cv2.recoverPose(E, pts1, pts2, K)
        return R, t

    @abstractmethod
    def get_features(self, frame:np.ndarray)-> tuple:
        pass

    @abstractmethod
    def match_features(self, desc1:list, desc2:list): 
        pass

    # @timit
    def estimate_trajectory(self, R, t):
        self.__P_next[:-1, :-1] = R
        self.__P_next[:-1, -1:] = t
        P_next_inv = np.linalg.inv(self.__P_next)
        self.__P = self.__P @ P_next_inv
        cam_pose = self.__P[:3, 3]
        self.__trajectory.append(cam_pose)
        return cam_pose

    @property
    def trajectory(self):
        return np.array(self.__trajectory[:])

    @timit
    def __call__(self, frame):
        self.__pair_buffer.append(frame)
        kpts, des = self.get_features(frame)
        self.__kpts_buffer.append(kpts)
        self.__des_buffer.append(des)
        if len(self.__pair_buffer) == 2:
            matched_pts = self.match_features(*self.__des_buffer)
            print([match.distance for match in matched_pts[:10]])
            R, t = MonoCamVisualOdom.estimate_motion(matched_pts, *self.__kpts_buffer, self.__K)
            trajectory_point = self.estimate_trajectory(R, t)
            self.__pair_buffer.pop(0)
            self.__kpts_buffer.pop(0)
            self.__des_buffer.pop(0)
            return trajectory_point


class SiftOdom(MonoCamVisualOdom):
    SIFT_N_FEATURES = 50
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
        sortedMatches = sorted(matched_pts, key=lambda x: x.distance)[:15]
        return sortedMatches

 
class OrbOdom(MonoCamVisualOdom):
    ORB_N_FEATURES = 500
    ORB_EDGE_THRESHOLD = 15

    def __init__(self, *args, **kwargs):
        super(OrbOdom, self).__init__(*args, **kwargs)
        self._matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        self._orb = cv2.ORB_create(
            nfeatures=kwargs.get('nfeatures', OrbOdom.ORB_N_FEATURES), 
            edgeThreshold=kwargs.get('edgeThreshold', OrbOdom.ORB_EDGE_THRESHOLD), 
        )

    def get_features(self, frame:np.uint8):
        kpts, des = self._orb.detectAndCompute(frame, None)
        return kpts, des

    def match_features(self, desc1:list, desc2:list):
        matched_pts = self._matcher.match(desc1, desc2)
        sortedMatches = sorted(matched_pts, key=lambda x: x.distance)[:250]
        return sortedMatches