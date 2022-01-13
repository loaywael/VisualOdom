#!/usr/bin/python3
import numpy as np
from scipy import linalg
import scipy

class PinHoleCamera:
    def __init__(
        self, imgW:int, imgH:int,
        P:np.float32=None,
        K:np.float32=None, 
        R:np.float32=np.eye(3),
        t:np.float32=np.zeros(3), 
        D:np.float32=None, 
    ) -> None:
        '''
        Params
        ------
        P : camera projection matrix
        K : camera intrinsics matrix
        R : camera rotation matrix
        t : camera translation vector
        D : camera distortion matrix
        '''
        self._imgW = imgW   # image width
        self._imgH = imgH   # image height
        self._imgSize = np.float32([self._imgW, imgH])
        self._D = D         # distortion
        self._setExtrinsicMtx(R, t)
        if isinstance(P, (np.ndarray)):
            self.setProjectionMtx(P)
        elif isinstance(K, (np.ndarray)): 
            self.setIntrinsicMtx(K)
        else: raise RuntimeError('(K, TF) and (P) enter either one')

    @property
    def D(self): 
        return self._D

    @property
    def P(self): 
        return self._P

    @property
    def K(self): 
        return self._K

    @property
    def TF(self): 
        return self._TF

    @property
    def fov(self):
        # (horizontal, vertical) FoV
        return np.rad2deg(np.arctan2((self._imgSize/2), np.abs(self._f)))*2  

    @property
    def center(self):
        return np.dot(self._R.T, -self._t)           # camera center 1x3

    def _setIntrinsicMtx(self, K):
        if K.shape != (3, 3):
            raise RuntimeError('Invalid matrix shape: K -> (3, 3) matrix')
        self._K = K                         # intrinsic matrix 3x3
        self._f = self._K[[0, 1], [0, 1]]   # focal length (fx, fy)
        self._s = self._K[0, 1]             # pixel skew
        self._c = self._K[:2, 2]            # image optical center

    def _setExtrinsicMtx(self, R, t):
        if (R.shape != (3, 3)) and (t.shape != (3, 3)):
            raise RuntimeError('Invalid matrix shape: R -> (3, 3) t -> (1, 3)')
        self._R = R                     # rotation matrix 3x3
        self._t = t                     # translation vector 1x3
        self._TF = np.eye(4)            # extrinsic matrix 4x4
        self._TF[:3, :3] = self._R
        self._TF[:3, 3] = self._t

    def setExtrinsicMtx(self, R, t):
        '''Updates the Camera extrinsic and projection matrices'''
        self._setExtrinsicMtx(R, t)
        self.setProjectionMtx(K=self._K, R=R, t=t)

    def setIntrinsicMtx(self, K):
        '''Updates the Camera intrinsic and projection matrices'''
        self._setIntrinsicMtx(K)
        self.setProjectionMtx(K=K, R=self._R, t=self._t)

    def setProjectionMtx(self, P=None, K=None, R=None, t=None):
        '''Updates the Camera projection matrix'''
        if P is not None:
            K, R, t = PinHoleCamera._factorProjectionMtx(P)
        self._setIntrinsicMtx(K)
        self._setExtrinsicMtx(R, t)
        self._P = np.dot(self._K, self._TF[:3])         # projection matrix 3x4

    @staticmethod
    def _factorProjectionMtx(P):
        '''Decompose the Camera projection matrix'''
        if P.shape != (3, 4):
            raise RuntimeError('Invalid matrix shape: P -> (3, 4) matrix')
        M = P[:3, :3]
        K, R = scipy.linalg.rq(M)
        T = np.diag(np.sign(np.diag(K)))
        if np.linalg.det(T) < 0:
            T[1, 1] *= -1
        K = np.dot(K, T)
        R = np.dot(T, R)
        t = np.dot(np.linalg.inv(K), P[:, 3])
        return K, R, t
        

class Camera(PinHoleCamera):
    @staticmethod
    def _projectPoints(X:np.float32, P:np.float32) -> np.float32:
        '''Projects 3D points on the Camera plane'''
        if isinstance(X, (np.ndarray)):
            assert (len(X.shape) == 2) and (X.shape[0] == 3)
            homoPoints = np.insert(X, 3, 1, axis=0)
            uvzPoints = np.dot(P, homoPoints)
            uvzPoints[:2, :] /= uvzPoints[2:3, :] + 1e-6
            return uvzPoints    # uvz-points (3, m)

    def _cropFramePoints(self, points:np.ndarray) -> np.bool8:
        '''Crops points outside/behind the Camera image plane'''
        xyzPoints = points.T    # (m, 3)
        xyzPoints = xyzPoints[xyzPoints[:, 2] > 0]    # remove behind camera
        U, V = xyzPoints[:, 0], xyzPoints[:, 1]       # (3, m)
        inFrameWidthMask = np.logical_and(U > 0, U < self._imgW)
        inFrameHeightMask = np.logical_and(V > 0, V < self._imgH)
        inFrameMask = np.logical_and(inFrameWidthMask, inFrameHeightMask)
        return xyzPoints[inFrameMask].T     # (3, m)

    def project(self, X:np.float32) -> np.float32:
        '''Projects 3D points on the Camera image plane and applies cropping'''
        projPoints = Camera._projectPoints(X, self._P)
        return self._cropFramePoints(projPoints)

    def spin(self, angle:float):
        '''Rotates Camera view around camera y-axis'''
        theta = np.deg2rad(angle)
        cos = np.cos(theta)
        sin = np.sin(theta)
        rotationAlongZ = np.array([
            [cos, -sin, 0],
            [sin, cos, 0],
            [0, 0, 1]
        ])
        self._P[:3, :3] = np.dot(self._P[:3, :3], rotationAlongZ)

    def view(self, scale:float=0.25) -> dict:
        '''Returns Camera View Mesh relative to the camera coordinates from Top-Left-CCW'''
        aspect = self._imgW / self._imgH    # view aspect ratio
        w, h = 1, 1/aspect                  # view size (w, h)
        depth = aspect*0.3*scale                # view virtual focal length
        viewPts = np.array([
            [0, 0, 0, 1],
            [-scale*w, -scale*h, depth, 1],
            [-scale*w, scale*h, depth, 1],
            [scale*w, scale*h, depth, 1],
            [scale*w, -scale*h, depth, 1]
        ])
        viewLines = [[0, i] for i in range(1, len(viewPts))]
        viewLines += [[i, i+1] for i in range(1, len(viewPts)-1)] + [[4, 1]]
        viewPoints = (self._TF @ viewPts.T)[:3].T    # camera -> world
        return dict(points=viewPoints, lines=viewLines, center=self.center)
