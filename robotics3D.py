from math import atan2, sin, cos, sqrt
import numpy as np


# quaternion_mul(q1,q2) takes two unit quaternions and returns their product
# (Quaternion Multiplication)
# ====>> N O T E : EXCLUSIVELY FOR QUATERNIONS WHICH REPRESENT ROTATION.
# THIS VERSION ENFORCES THE LAST ELEMENT OF THE VECTOR TO BE POSITIVE
#
# IF USING FOR REGULAR VECTORS, USE QUAT_VEC_MUL
def quat_mul(q1, q2):
    q1_matr = np.empty(shape=(4, 4))
    q1_matr[0][0] = q1[3]
    q1_matr[0][1] = q1[2]
    q1_matr[0][2] = -q1[1]
    q1_matr[0][3] = q1[0]

    q1_matr[1][0] = -q1[2]
    q1_matr[1][1] = q1[3]
    q1_matr[1][2] = q1[0]
    q1_matr[1][3] = q1[1]

    q1_matr[2][0] = q1[1]
    q1_matr[2][1] = -q1[0]
    q1_matr[2][2] = q1[3]
    q1_matr[2][3] = q1[2]

    q1_matr[3][0] = -q1[0]
    q1_matr[3][1] = -q1[1]
    q1_matr[3][2] = -q1[2]
    q1_matr[3][3] = q1[3]

    q = q1_matr.dot(q2)

    if q[3] < 0:
        q = -q

    return q / np.linalg.norm(q)


# quaternions_to_rot_matrix(q) takes a unit quaternion and returns the corresponding rotational matrix
def quat2rot(q):
    a = np.empty(shape=(3, 3))
    a[0][0] = q[0] ** 2 - q[1] ** 2 - q[2] ** 2 + q[3] ** 2
    a[0][1] = 2 * (q[0] * q[1] + q[2] * q[3])
    a[0][2] = 2 * (q[0] * q[2] - q[1] * q[3])

    a[1][0] = 2 * (q[0] * q[1] - q[2] * q[3])
    a[1][1] = -q[0] ** 2 + q[1] ** 2 - q[2] ** 2 + q[3] ** 2
    a[1][2] = 2 * (q[1] * q[2] + q[0] * q[3])

    a[2][0] = 2 * (q[0] * q[2] + q[1] * q[3])
    a[2][1] = 2 * (q[1] * q[2] - q[0] * q[3])
    a[2][2] = -q[0] ** 2 - q[1] ** 2 + q[2] ** 2 + q[3] ** 2

    return a


# takes a rotational matrix R and returns the roll, pitch and yaw angle
def rot2rpy(r):
    pitch = atan2(-r[1][0], (r[0][0] ** 2 + r[1][0] ** 2) ** .5)

    if abs(cos(pitch)) > 1.0e-12:
        yaw = atan2(r[1][0] / cos(pitch), r[0][0] / cos(pitch))
        roll = atan2(r[2][1] / cos(pitch), r[2][2] / cos(pitch))
    elif sin(pitch) > 0:
        yaw = 0
        roll = atan2(r[0][1], r[1][1])
    else:
        yaw = 0
        roll = atan2(-r[0][1], r[1][1])

    return np.array([roll, pitch, yaw])


# quaternion_inv(p) takes a unit quaternion p and returns its inverse
def quat_inv(p):
    q = np.empty(shape=4)
    q[0] = -p[0]
    q[1] = -p[1]
    q[2] = -p[2]
    q[3] = p[3]
    return q


# ROTVEC Rotation about arbitrary axis
#
# 	TR = AA2ROT(V, THETA)
#
# Returns a homogeneous transformation representing a rotation of THETA 
# about the vector V.
#
# See also: ROTX, ROTY, ROTZ.
def aa2rot(v, t):
    v /= np.linalg.norm(v)
    ct = cos(t)
    st = sin(t)
    vt = 1 - ct
    r = np.array([[ct, -v[2] * st, v[1] * st],
                  [v[2] * st, ct, -v[0] * st],
                  [-v[1] * st, v[0] * st, ct]])
    return np.outer(v, v * vt) + r


# converts a rotational matrix to a unit quaternion, according to JPL
# procedure (Breckenridge Memo)
# TODO: Extend to the full version
def rot2quat(r):
    t = r.trace()

    #    maxpivot = np.argmax([r[0][0], r[1][1], r[2][2], t])

    q = np.empty(shape=4)
    q[0] = sqrt((1 + 2 * r[0][0] - t) / 4)
    q[1:] = 1 / (4 * q[0]) * np.array([r[0][1] + r[1][0], r[0][2] + r[2][0], r[1][2] - r[2][1]])

    return q