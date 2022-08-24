#!/usr/bin/env python3
import numpy as np # Import Numpy

def DH(d, theta, a, alpha):
    '''
        Function builds elementary Denavit-Hartenberg transformation matrices 
        and returns the transformation matrix resulting from their multiplication.

        Arguments:
        d (double): displacement along Z-axis
        theta (double): rotation around Z-axis
        a (double): displacement along X-axis
        alpha (double): rotation around X-axis

        Returns:
        (Numpy array): composition of elementary DH transformations
    '''
    # 1. Build matrices representing elementary transformations (based on input parameters).
    # 2. Multiply matrices in the correct order (result in T).
    
    tran_z  = np.array([[1, 0, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, 1, d],
                        [0, 0, 0, 1]
                       ])

    rot_z = np.array([[np.cos(theta), -np.sin(theta),   0, 0],
                      [np.sin(theta),  np.cos(theta),    0, 0],
                      [0,                0,              1, 0],
                      [0,                0,              0, 1]
                     ])

    tran_x  = np.array([[1, 0, 0, a],
                        [0, 1, 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]
                       ])

    rot_x   = np.array([[1, 0, 0, 0],
                        [0, np.cos(alpha),  -np.sin(alpha), 0],
                        [0, np.sin(alpha),  np.cos(alpha),  0],
                        [0, 0,              0,              1]
                       ])
    
    T = tran_z @ rot_z @ tran_x @ rot_x
    return T

def kinematics(d, theta, a, alpha, Tb=np.eye(4)):
    '''
        Functions builds a list of transformation matrices, for a kinematic chain,
        descried by a given set of Denavit-Hartenberg parameters. 
        All transformations are computed from the base frame.

        Arguments:
        d (list of double): list of displacements along Z-axis
        theta (list of double): list of rotations around Z-axis
        a (list of double): list of displacements along X-axis
        alpha (list of double): list of rotations around X-axis

        Returns:
        (list of Numpy array): list of transformations along the kinematic chain (from the base frame)
    '''
    
    T = [Tb]
    
    for i in range(len(d)):
        temp = T[-1] @ DH(d[i], theta[i], a[i], alpha[i])
        T.append(temp)


    # T = [np.eye(4)] # Base transformation
    # For each set of DH parameters:
    # 1. Compute the DH transformation matrix.
    # 2. Compute the resulting accumulated transformation from the base frame.
    # 3. Append the computed transformation to T.
    return T

# Inverse kinematics
def jacobian(T, revolute):
    '''
        Function builds a Jacobian for the end-effector of a robot,
        described by a list of kinematic transformations and a list of joint types.

        Arguments:
        T (list of Numpy array): list of transformations along the kinematic chain of the robot (from the base frame)
        revolute (list of Bool): list of flags specifying if the corresponding joint is a revolute joint

        Returns:
        (Numpy array): end-effector Jacobian
    '''
    # 1. Initialize J and O.
    # 2. For each joint of the robot
    #   a. Extract z and o.
    #   b. Check joint type.
    #   c. Modify corresponding column of J.
    J = np.empty((6,len(revolute)))
    O_n = T[-1][0:3,3]
    
    for i in range(1,len(T)):
        Z = T[i-1][0:3,2]
        O = T[i-1][0:3,3]
        if revolute[i-1]:
            temp1 = np.cross(Z, O_n-O)
            temp2 = Z
        else:
            temp1 = Z
            temp2 = [0,0,0]

        temp = np.array([temp1, temp2]).reshape(6)
        J[:,i-1] = temp
    return J

# Damped Least-Squares
def DLS(A, damping, W=0):
    '''
        Function computes the damped least-squares (DLS) solution to the matrix inverse problem.

        Arguments:
        A (Numpy array): matrix to be inverted
        damping (double): damping factor
        W (Numpy array): weighting matrix

        Returns:
        (Numpy array): inversion of the input matrix
    '''
    if type(W) is int:
        W=np.eye(A.shape[1])
    Winv = np.linalg.inv(W)
    return Winv@A.T@np.linalg.inv(A@Winv@A.T + (damping**2)*np.eye(A.shape[0]))# Implement the formula to compute the DLS of matrix A.

# Extract characteristic points of a robot projected on X-Y plane
def robotPoints2D(T):
    '''
        Function extracts the characteristic points of a kinematic chain on a 2D plane,
        based on the list of transformations that describe it.

        Arguments:
        T (list of Numpy array): list of transformations along the kinematic chain of the robot (from the base frame)
    
        Returns:
        (Numpy array): an array of 2D points
    '''
    P = np.zeros((2,len(T)))
    for i in range(len(T)):
        P[:,i] = T[i][0:2,3]
    return P