import numpy as np
import cv2
import math
import open3d
import os

from tk3dv.nocstools import datastructures as ds 

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

BILATERAL_DIAM = 7
BILATERAL_SIGMA = 90
MEDIAN_KSIZE = 3


def nocs2pc(nocs_list):
        ''' Turns a tuple of NOCS maps into a combined point cloud '''
        nocs_pc = []
        for nocs_map in nocs_list:
            nocs = ds.NOCSMap(nocs_map)
            nocs_pc.append(nocs.Points)
        nocs_pc = np.concatenate(nocs_pc, axis=0)
        return nocs_pc

def read_nocs_map(path):
        nocs_map = cv2.imread(path, -1)
        nocs_map = nocs_map[:, :, :3] # Ignore alpha if present
        nocs_map = cv2.cvtColor(nocs_map, cv2.COLOR_BGR2RGB)
        return nocs_map

def write_nocs_map(nocs_map, path):
        nocs_map = cv2.cvtColor(nocs_map, cv2.COLOR_RGB2BGR)
        cv2.imwrite(path, nocs_map)

def filter_nocs_map(src_map, map_filter):
    out_nocs_map = src_map
    if map_filter == 'median':
        out_nocs_map = cv2.medianBlur(out_nocs_map, MEDIAN_KSIZE)
    if map_filter == 'bilateral':
        out_nocs_map = cv2.bilateralFilter(out_nocs_map, BILATERAL_DIAM, BILATERAL_SIGMA, BILATERAL_SIGMA)
    return out_nocs_map


def dict2np(vec_dict):
    if len(vec_dict) == 3:
        return np.array([vec_dict['x'], vec_dict['y'], vec_dict['z']], dtype=np.float)
    elif len(vec_dict) == 4:
        return np.array([vec_dict['w'], vec_dict['x'], vec_dict['y'], vec_dict['z']], dtype=np.float)

def prune_pc(pc, max_pts):
    ''' Reduces number of points in the point cloud by random sampling. '''
    if pc.shape[0] > max_pts:
        keep_inds = np.random.choice(pc.shape[0], size=max_pts, replace=False)
        pc = pc[keep_inds]
    return pc


def save_pc_single(pc, out_path, meta_info):
    ''' Saves a point cloud to a npy file at the given path '''
    save_f = meta_info[0] + '_' + meta_info[1] + '_' + str(meta_info[2]).zfill(3)
    np.save(os.path.join(out_path, save_f), pc)

def save_pc_multi(pc, out_path, meta_info):
    ''' Saves a point cloud to a npy file at the given path '''
    save_f = meta_info[0] + '_' + meta_info[1]
    np.save(os.path.join(out_path, save_f), pc)

#
# utilities from DPC code
#
def ypr_from_campos(cx, cy, cz):
    camDist = math.sqrt(cx * cx + cy * cy + cz * cz)
    cx = cx / camDist
    cy = cy / camDist
    cz = cz / camDist
    t = math.sqrt(cx * cx + cy * cy)
    tx = cx / t
    ty = cy / t
    yaw = math.acos(tx)
    if ty > 0:
        yaw = 2 * math.pi - yaw

    roll = 0
    pitch = math.asin(cz)

    return yaw, pitch, roll

def ypr_from_campos_blender(pos):
    yaw, pitch, roll = ypr_from_campos(pos[0], pos[1], pos[2])
    yaw = yaw + np.pi
    return yaw, pitch, roll


def axis_angle_quaternion(angle, axis):
    c = math.cos(angle / 2)
    s = math.sin(angle / 2)
    q = np.zeros(4)
    q[0] = c
    q[1:4] = s * axis
    return q

def quaternion_multiply(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
    z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2
    return np.array([w, x, y, z])

def quaternion_conjugate(q):
    """Compute the conjugate of q, i.e. [q.w, -q.x, -q.y, -q.z]."""
    return q * np.array([1.0, -1.0, -1.0, -1.0])

def quaternionFromYawPitchRoll(yaw, pitch, roll):
    # reverse transformation is ypr = quaternion2euler(quat)
    q_yaw = axis_angle_quaternion(yaw, np.array([0, 1, 0]))
    q_pitch = axis_angle_quaternion(pitch, np.array([0, 0, 1]))
    q_roll = axis_angle_quaternion(roll, np.array([1, 0, 0]))
    return quaternion_multiply(q_roll, quaternion_multiply(q_pitch, q_yaw))

def quaternion_from_campos(cam_pos):
    yaw, pitch, roll = ypr_from_campos_blender(cam_pos)
    return quaternionFromYawPitchRoll(yaw, pitch, roll)

def open3d_icp(src, trgt, init_rotation=np.eye(3, 3)):
    source = open3d.PointCloud()
    source.points = open3d.Vector3dVector(src)

    target = open3d.PointCloud()
    target.points = open3d.Vector3dVector(trgt)

    init_rotation_4x4 = np.eye(4, 4)
    init_rotation_4x4[0:3, 0:3] = init_rotation

    threshold = 0.2
    reg_p2p = open3d.registration_icp(source, target, threshold, init_rotation_4x4,
                                    open3d.TransformationEstimationPointToPoint())

    return reg_p2p

def viz_pcs(pcs, path):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for pc in pcs:
                ax.scatter(pc[:, 0], pc[:, 2], pc[:, 1], marker='o')
        ax.set_xlabel('X')
        ax.set_ylabel('Z')
        ax.set_zlabel('Y')
        ax.set_xlim(-0.5, 0.5)
        ax.set_ylim(-0.5, 0.5)
        ax.set_zlim(-0.5, 0.5)
        plt.savefig(path)
        plt.close(fig)

def normalize_pc(pc):
        ''' Normalize so diagonal of tight bounding box is 1 '''
        diagonal_len_sqr = get_pc_diag(pc)**2
        norm_pc = pc * np.sqrt(1.0 / diagonal_len_sqr)
        return norm_pc

def get_pc_diag(pc):
        xwidth = np.amax(pc[:,0]) - np.amin(pc[:,0])
        ywidth = np.amax(pc[:,1]) - np.amin(pc[:,1])
        zwidth = np.amax(pc[:,2]) - np.amin(pc[:,2])
        diagonal_len = np.sqrt(xwidth**2 + zwidth**2 + ywidth**2)
        return diagonal_len

def as_rotation_matrix(q):
    """Calculate the corresponding rotation matrix.

    See
    http://www.euclideanspace.com/maths/geometry/rotations/conversions/quaternionToMatrix/
    """
    # helper functions
    def diag(a, b):  # computes the diagonal entries,  1 - 2*a**2 - 2*b**2
        return 1 - 2 * np.power(a, 2) - 2 * np.power(b, 2)

    def tr_add(a, b, c, d):  # computes triangle entries with addition
        return 2 * a * b + 2 * c * d

    def tr_sub(a, b, c, d):  # computes triangle entries with subtraction
        return 2 * a * b - 2 * c * d

    w, x, y, z = q / np.linalg.norm(q)
    m = [[diag(y, z), tr_sub(x, y, z, w), tr_add(x, z, y, w)],
         [tr_add(x, y, z, w), diag(x, z), tr_sub(y, z, x, w)],
         [tr_sub(x, z, y, w), tr_add(y, z, x, w), diag(x, y)]]
    return np.stack([np.stack(m[i], axis=-1) for i in range(3)], axis=-2)


'''
The MIT License (MIT)

Copyright (c) 2014 Tolga Birdal, Eldar Insafutdinov

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Website: https://github.com/tolgabirdal/averaging_quaternions
'''
def quatWAvgMarkley(Q):
    """
    ported from the original Matlab implementation at:
    https://www.mathworks.com/matlabcentral/fileexchange/40098-tolgabirdal-averaging_quaternions

    by Tolga Birdal
    Q is an Mx4 matrix of quaternions. weights is an Mx1 vector, a weight for
    each quaternion.
    Qavg is the weightedaverage quaternion
    This function is especially useful for example when clustering poses
    after a matching process. In such cases a form of weighting per rotation
    is available (e.g. number of votes), which can guide the trust towards a
    specific pose. weights might then be interpreted as the vector of votes
    per pose.
    Markley, F. Landis, Yang Cheng, John Lucas Crassidis, and Yaakov Oshman.
    "Averaging quaternions." Journal of Guidance, Control, and Dynamics 30,
    no. 4 (2007): 1193-1197.
    """
    # Form the symmetric accumulator matrix
    A = np.zeros((4, 4))
    M = Q.shape[0]
    weights = np.ones(M)

    wSum = 0

    for i in range(M):
        q = Q[i, :]
        q = np.expand_dims(q, -1)
        w_i = weights[i]
        A = w_i * np.matmul(q, q.transpose()) + A  # rank 1 update
        wSum = wSum + w_i

    # scale
    A = 1.0 / wSum * A

    # Get the eigenvector corresponding to largest eigen value
    w, v = np.linalg.eig(A)
    ids = np.argsort(w)
    idx = ids[-1]
    q_avg = v[:, idx]
    if q_avg[0] < -0:
        q_avg *= -1.0
    return q_avg
