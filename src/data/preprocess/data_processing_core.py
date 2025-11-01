"""
    from: http://phi-ai.buaa.edu.cn/Gazehub/rectification/#tutorial
"""
import math

import cv2
import numpy as np


class norm:
    def __init__(
        self, center, gazetarget, headrotvec, imsize, camparams, newfocal=960, newdistance=600
    ):
        try:
            self.center = np.array(center)
            self.headrotvec = np.array(headrotvec)
            self.target = np.array(gazetarget)
            self.imsize = np.array(imsize)
            self.cameraparams = np.array(camparams)
        except:
            print("There are some errors in inputs")
            exit()

        self.newfocal = newfocal
        self.newdistance = newdistance
        self.curdistance = np.linalg.norm(self.center)

        self.__assertion()  # To make sure the correctness of inputs.

        if self.headrotvec.shape == (3,):
            self.headrotvec = cv2.Rodrigues(self.headrotvec)[0]

        self.__ParamsCalculate()  # To calculate and save some required params.

    def __assertion(self):
        assert self.center.shape == (3,), print("Center's Pattern Must Be [x, y, z].")
        assert self.headrotvec.shape == (3,) or self.headrotvec.shape == (3, 3), print(
            "rotvec's Patttern Must Be [x, y, z] or 3*3 Mat."
        )
        assert self.target.shape == (3,), print("Target's Pattern Must Be [x, y, z].")
        assert self.imsize.shape == (2,), print("Imsize's Pattern Must Be [x, y].")
        assert self.cameraparams.shape == (3, 3), print("CamParams's Pattern Must Be 3*3 Mat.")
        assert type(self.newfocal) == int or type(self.newfocal) == float, print(
            "New focal must be int or float."
        )
        assert type(self.newdistance) == int or type(self.newdistance) == float, print(
            "New distance must be int or float."
        )

    def __ParamsCalculate(self):
        #  Matrix: S, R, M=S*R, C_n, W=C_n*M*(C)^-1
        #  Also provide gaze vec and head rotation matrix
        self.S_mat = np.array([[1, 0, 0], [0, 1, 0], [0, 0, self.newdistance / self.curdistance]])
        xaxis = self.headrotvec[:, 0]
        z = self.center / self.curdistance
        y = np.cross(z, xaxis)
        y = y / np.linalg.norm(y)
        x = np.cross(y, z)
        x = x / np.linalg.norm(x)

        self.R_mat = np.array([x, y, z])
        self.C_mat = np.array(
            [
                [self.newfocal, 0, self.imsize[0] / 2],
                [0, self.newfocal, self.imsize[1] / 2],
                [0, 0, 1],
            ]
        )
        self.M_mat = np.dot(self.S_mat, self.R_mat)
        self.W_mat = np.dot(np.dot(self.C_mat, self.M_mat), np.linalg.inv(self.cameraparams))
        self.gaze = self.target - self.center

    def GetParams(self):
        rvec = cv2.Rodrigues(self.R_mat)[0].flatten()
        svec = np.diagonal(self.S_mat)
        return rvec, svec

    def GetImage(self, image):
        self.im = cv2.warpPerspective(
            image, self.W_mat, (int(self.imsize[0]), int(self.imsize[1]))
        )
        return self.im

    def GetCoordinate(self, coordinate):
        coordinate = np.reshape(coordinate, (3, 1))
        return np.dot(self.M_mat, coordinate).flatten()

    def GetGaze(self, scale=True):
        if scale:
            gaze = np.dot(self.M_mat, self.gaze)
            gaze = gaze / np.linalg.norm(gaze)
        else:
            gaze = np.dot(self.R_mat, self.gaze)
            gaze = gaze / np.linalg.norm(gaze)

        return gaze

    def GetHeadRot(self, vector=True):
        if vector:
            return cv2.Rodrigues(np.dot(self.M_mat, self.headrotvec))[0].T[0]
        else:
            return np.dot(self.M_mat, self.headrotvec)

    def GetNewPos(self, position):
        try:
            pos = np.array(position).astype("float32")
            assert pos.shape == (2,), print("GetNewPos need 2 dim vector")
            pos = np.append(pos, 1)
            result = np.dot(pos, self.W_mat.T)
            return np.array([result[0] / result[2], result[1] / result[2]])
        except:
            print("Error in GetNewPos")
            exit()

    def CropEye(self, lcorner, rcorner):
        try:
            self.im
        except:
            print("There is no image, please use GetImage first.")

        x, y = list(zip(lcorner, rcorner))

        center_x = np.mean(x)
        center_y = np.mean(y)

        width = np.abs(x[0] - x[1]) * 1.5
        times = width / 60
        height = 36 * times

        x1 = [max(center_x - width / 2, 0), max(center_y - height / 2, 0)]
        x2 = [min(x1[0] + width, self.imsize[0]), min(x1[1] + height, self.imsize[1])]
        im = self.im[int(x1[1]) : int(x2[1]), int(x1[0]) : int(x2[0])]
        im = cv2.resize(im, (60, 36))
        return im

    def CropEyeWithCenter(self, center):
        try:
            self.im
        except:
            print("There is no image, please use GetImage first.")
        center_x = center[0]
        center_y = center[1]
        width = 60 * 1.2
        height = 36 * 1.2

        x1 = [max(center_x - width / 2, 0), max(center_y - height / 2, 0)]
        x2 = [min(x1[0] + width, self.imsize[0]), min(x1[1] + height, self.imsize[1])]
        im = self.im[int(x1[1]) : int(x2[1]), int(x1[0]) : int(x2[0])]
        im = cv2.resize(im, (60, 36))
        return im


def GazeTo2d(gaze):
    yaw = np.arctan2(-gaze[0], -gaze[2])
    pitch = np.arcsin(-gaze[1])
    return np.array([pitch, yaw])


def GazeTo3d(gaze):
    x = -np.cos(gaze[1]) * np.sin(gaze[0])
    y = -np.sin(gaze[1])
    z = -np.cos(gaze[1]) * np.cos(gaze[0])
    return np.array([x, y, z])


def HeadTo2d(head):
    assert np.array(head).shape == (
        3,
    ), f"The shape of headrotmatrix must be (3,), which is {np.array(head).shape} currently"
    M = cv2.Rodrigues(head)[0]
    # print(M, end="-----")
    vec = M[:, 2]
    pitch = np.arcsin(vec[1])
    yaw = np.arctan2(vec[0], vec[2])
    return np.array([pitch, yaw])


def GazeFlip(gaze):
    newgaze = np.zeros([3])
    newgaze[0] = -gaze[0]
    newgaze[1] = gaze[1]
    newgaze[2] = gaze[2]
    return newgaze


def HeadFlip(head):
    rot_vec = np.array(head)
    assert head.shape == (
        3,
    ), f"The shape of headrotvec must be (3,), which is {head.shape} currently."

    rot_mat = cv2.Rodrigues(rot_vec)[0]
    z = rot_mat[:, 2]
    y = rot_mat[:, 1]

    z[0] = -z[0]
    x = np.cross(y, z)

    newrot_mat = np.array([x, y, z])
    newrot_vec = cv2.Rodrigues(rot_mat)[0].T[0]
    return newrot_vec


def EqualizeHist(img):
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return cv2.equalizeHist(img)


def Euler2RotMat(theta, format="degree"):
    """
    Calculates Rotation Matrix given euler angles.
    :param theta: 1-by-3 list [rx, ry, rz] angle in degree
    :return:
    RPY, the object will be rotated with the order of [rx, ry, rz]
    """
    if format == "degree":
        theta = [i * math.pi / 180.0 for i in theta]

    R_x = np.array(
        [
            [1, 0, 0],
            [0, math.cos(theta[0]), -math.sin(theta[0])],
            [0, math.sin(theta[0]), math.cos(theta[0])],
        ]
    )

    R_y = np.array(
        [
            [math.cos(theta[1]), 0, math.sin(theta[1])],
            [0, 1, 0],
            [-math.sin(theta[1]), 0, math.cos(theta[1])],
        ]
    )

    R_z = np.array(
        [
            [math.cos(theta[2]), -math.sin(theta[2]), 0],
            [math.sin(theta[2]), math.cos(theta[2]), 0],
            [0, 0, 1],
        ]
    )
    R = np.dot(R_z, np.dot(R_y, R_x))
    return R
