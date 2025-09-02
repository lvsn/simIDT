"""
    Transforms contain utility functions to manipulate pointclouds

    date : 2016-03-01
"""
from scipy.stats import cauchy

__author__ = "Mathieu Garon"
__version__ = "0.0.1"

import numpy as np
import random
import math

try:
    import angles as ea
except ImportError:
    import pem.utils.angles as ea


class Transform:
    def __init__(self):
        self.matrix = np.eye(4, dtype=np.float32)

    def set_translation(self, x, y, z):
        self.matrix[0:3, 3] = [x, y, z]

    def set_rotation(self, x, y, z):
        self.matrix[0:3, 0:3] = ea.rodrigues(x, y, z)

    def translate(self, x=0, y=0, z=0, transform=None):
        """
        The translation is applied before the transforme
        TODO: this is not consistant...
        :param x:
        :param y:
        :param z:
        :param transform:
        :return:
        """
        if transform:
            new_transform = transform.copy()
        else:
            new_transform = Transform.from_parameters(x, y, z, 0, 0, 0)
        new_transform.combine(self)
        self.matrix = new_transform.matrix
        return self

    def rotate(self, x=0, y=0, z=0, transform=None, is_degree=False):
        if transform:
            self.combine(transform)
        else:
            self.combine(Transform.from_parameters(0, 0, 0, x, y, z, is_degree=False))
        return self

    @staticmethod
    def random(translation_range=(-1, 1), rotation_range=(-1, 1)):
        x = random.uniform(*translation_range)
        y = random.uniform(*translation_range)
        z = random.uniform(*translation_range)
        a = random.uniform(*rotation_range)
        b = random.uniform(*rotation_range)
        c = random.uniform(*rotation_range)
        return Transform.from_parameters(x, y, z, a, b, c)

    @staticmethod
    def random_cauchydist(translation_range=(-1, 1), rotation_range=(-1, 1), mean_t=0, sigma_t=1, mean_r=0, sigma_r=1):

        x = Transform.clipped_cauchy(translation_range[0], translation_range[1], mean_t, sigma_t)
        y = Transform.clipped_cauchy(translation_range[0], translation_range[1], mean_t, sigma_t)
        z = Transform.clipped_cauchy(translation_range[0], translation_range[1], mean_t, sigma_t)
        a = Transform.clipped_cauchy(rotation_range[0], rotation_range[1], mean_r, sigma_r)
        b = Transform.clipped_cauchy(rotation_range[0], rotation_range[1], mean_r, sigma_r)
        c = Transform.clipped_cauchy(rotation_range[0], rotation_range[1], mean_r, sigma_r)
        params = np.array([x, y, z, a, b, c])
        return Transform.from_parameters(*params)

    @staticmethod
    def clipped_cauchy(min, max, mean, sigma):
        rv = cauchy(mean, sigma)
        val = rv.rvs()
        while val < min or val > max:
            val = rv.rvs()
        return val

    @staticmethod
    def from_matrix(matrix):
        ret = Transform()
        ret.matrix = matrix
        return ret

    @staticmethod
    def scale(x, y, z):
        ret = Transform()
        ret.matrix[0, 0] = x
        ret.matrix[1, 1] = y
        ret.matrix[2, 2] = z
        return ret

    @staticmethod
    def lookAt(eye, center, up):
        ret = Transform()
        E = eye
        C = center
        U = up

        F = C - E
        F /= np.linalg.norm(F)
        S = np.cross(F, U)
        S /= np.linalg.norm(S)
        U = np.cross(S, F)

        mat = np.eye(4, dtype=np.float32)

        mat[0, :] = np.hstack([S, 0])
        mat[1, :] = np.hstack([U, 0])
        mat[2, :] = np.hstack([-F, 0])
        mat[0, 3] = -np.dot(S, E)
        mat[1, 3] = -np.dot(U, E)
        mat[2, 3] = np.dot(F, E)
        ret.matrix = mat
        return ret

    def to_parameters(self, isDegree=False, isQuaternion=False, rodrigues=True):
        x, y, z = self.matrix[0:3, 3]
        if rodrigues:
            rx, ry, rz = ea.rodrigues_inverse(self.matrix[0:3, 0:3])
        else:
            rx, ry, rz = ea.mat2euler(self.matrix[0:3, 0:3])
        if isDegree:
            rx = math.degrees(rx)
            ry = math.degrees(ry)
            rz = math.degrees(rz)
        ret = [x, y, z, rx, ry, rz]
        if isQuaternion:
            qx, qy, qz, qw = ea.euler2quat(x=rx, y=ry, z=rz)
            ret = [x, y, z, qx, qy, qz, qw]
        return np.array(ret)

    @staticmethod
    def from_parameters(x, y, z, euler_x, euler_y, euler_z, is_degree=False):
        ret = Transform()
        ret.set_translation(x, y, z)
        if is_degree:
            euler_x = math.radians(euler_x)
            euler_y = math.radians(euler_y)
            euler_z = math.radians(euler_z)
        ret.set_rotation(euler_x, euler_y, euler_z)
        return ret
    
    @staticmethod
    def transform_pts_batch(pts, R, t=None):
        # Source: Thomas Hodan, 
        # https://github.com/THU-DA-6D-Pose-Group/self6dpp/blob/d94eaf7045ff3f2d992f23735e7dc3011338a3fd/lib/pysixd/misc.py
        """
        Args:
            pts: (B,P,3)
            R: (B,3,3)
            t: (B,3,1)

        Returns:

        """
        bs = R.shape[0]
        n_pts = pts.shape[1]
        assert pts.shape == (bs, n_pts, 3)
        if t is not None:
            assert t.shape[0] == bs

        pts_transformed = R.view(bs, 1, 3, 3) @ pts.view(bs, n_pts, 3, 1)
        if t is not None:
            pts_transformed += t.view(bs, 1, 3, 1)
        return pts_transformed.squeeze(-1)  # (B, P, 3)

    @staticmethod
    def transform_pts_Rt(pts, R, t):
        # Source: Thomas Hodan, 
        # https://github.com/THU-DA-6D-Pose-Group/self6dpp/blob/d94eaf7045ff3f2d992f23735e7dc3011338a3fd/lib/pysixd/misc.py
        """Applies a rigid transformation to 3D points.

        :param pts: nx3 ndarray with 3D points.
        :param R: 3x3 ndarray with a rotation matrix.
        :param t: 3x1 ndarray with a translation vector.
        :return: nx3 ndarray with transformed 3D points.
        """
        assert pts.shape[1] == 3
        pts_t = R.dot(pts.T) + t.reshape((3, 1))
        return pts_t.T


    @property
    def shape(self):
        return self.matrix.shape

    @property
    def rotation(self):
        ret = Transform()
        ret.matrix[0:3, 0:3] = self.matrix[0:3, 0:3]
        return ret

    @property
    def translation(self):
        ret = Transform()
        ret.matrix[0:3, 3] = self.matrix[0:3, 3]
        return ret

    def inverse(self):
        ret = Transform()
        ret.matrix[0:3, 0:3] = self.matrix[0:3, 0:3].transpose()
        ret.matrix[0:3, 3] = -ret.matrix[0:3, 0:3].dot(self.matrix[0:3, 3])
        return ret

    def transpose(self):
        ret = Transform()
        ret.matrix = self.matrix.T
        return ret

    def dot(self, points):
        shape = points.shape
        if shape[1] == 3:
            # to homogeneous (stack layer of one)
            ones = np.ones((shape[0], 1))
            homogeneous = np.hstack((points, ones)).T
        elif shape[1] == 4:
            homogeneous = points.T
        else:
            raise ValueError("input array has to be of size 3 or in homogeneous coordinate, current size = " + str(shape))
        return self.matrix.dot(homogeneous).T[:, 0:3]

    def combine(self, transform, copy=False):
        ret_transform = self
        if not copy:
            self.matrix = self.matrix.dot(transform.matrix)
        else:
            new_matrix = self.matrix.dot(transform.matrix)
            ret_transform = Transform.from_matrix(new_matrix)
        return ret_transform

    def copy(self):
        ret = Transform()
        ret.matrix = self.matrix.copy()
        return ret

    def __getitem__(self, item):
        return self.matrix[item]

    def __setitem__(self, key, value):
        self.matrix[key] = value

    def __str__(self):
        params = self.to_parameters(isDegree=True)
        ret = ""
        ret += "x :" + str(params[0]) + ",\n"
        ret += "y :" + str(params[1]) + ",\n"
        ret += "z :" + str(params[2]) + ",\n"
        ret += "x :" + str(params[3]) + " degrees,\n"
        ret += "y :" + str(params[4]) + " degrees,\n"
        ret += "z :" + str(params[5]) + " degrees.\n"
        return ret

    def __repr__(self):
        return str(self.matrix)

    def __eq__(self, other):
        """Override the default Equals behavior"""
        if isinstance(other, self.__class__):
            return np.isclose(self.matrix, other.matrix).all()
        return False

    def __ne__(self, other):
        return not self.__eq__(other)
