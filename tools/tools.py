import numpy as np

def compute_homogeneous_transforms(x, y, z, alpha,beta, gamma):
    R = euler2rot([alpha, beta, gamma])
    T = np.zeros((4, 4))
    T[0:3, 0:3] = R
    T[3, 3] = 1
    T[0:3, 3] = np.array([x, y, z])
    return T

def quaternion2rot(Q):
    qw = Q[0]
    qx = Q[1]
    qy = Q[2]
    qz = Q[3]
    R = np.eye(3)
    R[0, 0] = 1 - 2 * qy**2 - 2 * qz**2
    R[0, 1] = 2 * qx * qy - 2 * qz * qw
    R[0, 2] = 2 * qx * qz + 2 * qy * qw
    R[1, 0] = 2 * qx * qy + 2 * qz * qw
    R[1, 1] = 1 - 2*qx**2 - 2*qz**2
    R[1, 2] = 2 * qy * qz - 2 * qx * qw
    R[2, 0] = 2 * qx * qz - 2 * qy * qw
    R[2, 1] = 2 * qy * qz + 2 * qx * qw
    R[2, 2] = 1 - 2 * qx**2 - 2 * qy**2
    return R

def transform_to_pose(T):
    tx = T[0, 3]
    ty = T[1, 3]
    tz = T[2, 3]
    R = T[0:3, 0:3]
    euler = rot2euler(R)
    return tx,ty,tz,euler

def rot2euler(R):

    th = np.abs(np.abs(R[0, 2])-1.0)
    R[0, 2] = min(R[0, 2], 1)
    R[0, 2] = max(R[0, 2], -1)

    if th > 0.0001:
        beta1 = np.arcsin(R[0, 2])
        beta2 = np.pi - beta1
        s1 = np.sign(np.cos(beta1))
        s2 = np.sign(np.cos(beta2))
        alpha1 = np.arctan2(-s1*R[1][2], s1*R[2][2])
        gamma1 = np.arctan2(-s1*R[0][1], s1*R[0][0])
        alpha2 = np.arctan2(-s2*R[1][2], s2*R[2][2])
        gamma2 = np.arctan2(-s2*R[0][1], s2*R[0][0])
    else:
        alpha1 = 0
        alpha2 = np.pi
        beta1 = np.arcsin(R[0, 2])
        if beta1 > 0:
            beta2 = np.pi/2
            gamma1 = np.arctan2(R[1][0], R[1][1])
            gamma2 = np.arctan2(R[1][0], R[1][1])-alpha2
        else:
            beta2 = -np.pi/2
            gamma1 = np.arctan2(-R[1][0], R[1][1])
            gamma2 = np.arctan2(-R[1][0], R[1][1])-alpha2

    e1 = normalize_angle([alpha1, beta1, gamma1])
    e2 = normalize_angle([alpha2, beta2, gamma2])
    return e1

def normalize_angle(eul):
    """
    Normalize angles in array to [-pi, pi]
    """
    e = []
    for i in range(len(eul)):
        e.append(np.arctan2(np.sin(eul[i]), np.cos(eul[i])))
    return e

def euler2rot(abg):
    calpha = np.cos(abg[0])
    salpha = np.sin(abg[0])
    cbeta = np.cos(abg[1])
    sbeta = np.sin(abg[1])
    cgamma = np.cos(abg[2])
    sgamma = np.sin(abg[2])
    Rx = np.array([[1, 0, 0], [0, calpha, -salpha], [0, salpha, calpha]])
    Ry = np.array([[cbeta, 0, sbeta], [0, 1, 0], [-sbeta, 0, cbeta]])
    Rz = np.array([[cgamma, -sgamma, 0], [sgamma, cgamma, 0], [0, 0, 1]])
    R = np.matmul(Rx, Ry)
    R = np.matmul(R, Rz)
    return R
