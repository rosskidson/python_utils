import numpy as np


def euler_to_quaternion(roll, pitch, yaw):

    qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
    qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
    qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)

    return [qx, qy, qz, qw]


def euler_to_quaternion_vec(vec):
    return euler_to_quaternion(vec[0], vec[1], vec[2])


def quaternion_to_euler(qx, qy, qz, qw):
    t0 = +2.0 * (qw * qx + qy * qz)
    t1 = +1.0 - 2.0 * (qx * qx + qy * qy)
    roll =  np.arctan2(t0, t1)

    t2 = +2.0 * (qw * qy - qz * qx)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch = np.arcsin(t2)

    t3 = +2.0 * (qw * qz + qx * qy)
    t4 = +1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = np.arctan2(t3, t4)

    return roll, pitch, yaw


def quaternion_to_euler_vec(vec):
    return quaternion_to_euler(vec[0], vec[1], vec[2], vec[3])


def restore_minimal_qt(qx, qy, qz):
    squared_norm = qx * qx + qy * qy + qz * qz
    qw = np.sqrt(1 - squared_norm)
    return (qx, qy, qz, qw)


def restore_minimal_qt_vec(q_vec):
    return restore_minimal_qt(q_vec[0], q_vec[1], q_vec[2])


def Rmat_to_quaternion(R):
    qw = 0.5 * np.sqrt(1 + R[0,0] + R[1,1] + R[2,2])
    qx = 1 /(4 * qw)*(R[2,1] - R[1,2])
    qy = 1 /(4 * qw)*(R[0,2] - R[2,0])
    qz = 1 /(4 * qw)*(R[1,0] - R[0,1])
    return qx, qy, qz, qw


def quaternion_to_Rmat(qx, qy, qz, qw):
    R =  np.matrix([[1 - 2*qy*qy - 2*qz*qz, 2*(qx*qy - qz*qw),     2*(qx*qz + qy*qw)],
                    [2*(qx*qy + qz*qw),     1 - 2*qx*qx - 2*qz*qz, 2*(qy*qz - qx*qw)],
                    [2*(qx*qz - qy*qw),     2*(qx*qw + qy*qz),     1 - 2*qx*qx - 2*qy*qy]])
    return R
