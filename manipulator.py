from kinematics import * # Includes numpy import

def jacobianLink(T, revolute, link): # Needed in Exercise 2
    '''
        Function builds a Jacobian for the end-effector of a robot,
        described by a list of kinematic transformations and a list of joint types.

        Arguments:
        T (list of Numpy array): list of transformations along the kinematic chain of the robot (from the base frame)
        revolute (list of Bool): list of flags specifying if the corresponding joint is a revolute joint
        link(integer): index of the link for which the Jacobian is computed

        Returns:
        (Numpy array): end-effector Jacobian
    '''
    # Code almost identical to the one from lab2_robotics...
    J = np.zeros((6,len(revolute)))
    O_n = T[link][0:3,3]
    for i in range(0,link):
        Z = T[i][0:3,2]
        O = T[i][0:3,3]
        if revolute[i]:
            temp1 = np.cross(Z, O_n-O)
            temp2 = Z
        else:
            temp1 = Z
            temp2 = [0,0,0]

        temp = np.array([temp1, temp2]).reshape(6)
        J[:,i] = temp
    return J

class Manipulator:
    '''
    Class representing a robotic manipulator.
    '''
    def __init__(self, d, theta, a, alpha, revolute):
        '''
            Constructor.

            Arguments:
            d (Numpy array): list of displacements along Z-axis
            theta (Numpy array): list of rotations around Z-axis
            a (Numpy array): list of displacements along X-axis
            alpha (Numpy array): list of rotations around X-axis
            revolute (list of Bool): list of flags specifying if the corresponding joint is a revolute joint
        '''
        self.d = d
        self.theta = theta
        self.a = a
        self.alpha = alpha
        self.revolute = revolute
        self.dof = len(self.revolute)
        self.q = np.zeros(self.dof).reshape(-1, 1)
        self.update(0.0, 0.0)


    def update(self, dq, dt):
        '''
        Method that updates the state of the robot.

        Arguments:
        dq (Numpy array): a column vector of joint velocities
        dt (double): sampling time
        '''
        self.q += dq * dt
        for i in range(len(self.revolute)):
            if self.revolute[i]:
                self.theta[i] = self.q[i]
            else:
                self.d[i] = self.q[i]
        self.T = kinematics(self.d, self.theta, self.a, self.alpha)

    
    def drawing(self):
        ''' 
        Method that returns the characteristic points of the robot.
        '''
        return robotPoints2D(self.T)

    
    def getEEJacobian(self):
        '''
        Method that returns the end-effector Jacobian.
        '''
        return jacobian(self.T, self.revolute)

    
    def getEETransform(self):
        '''
        Method that returns the end-effector transformation.
        '''
        return self.T[-1]
    def getLinkJacobian(self, link):
        '''
        Method that returns the link Jacobian.
        '''
        return jacobianLink(self.T, self.revolute, link)

    
    def getLinkTransform(self, link):
        '''
        Method that returns the link transformation.
        '''
        return self.T[link]


    def getJointPos(self, joint):
        '''
            Method that returns the position of a selected joint.

            Argument:
            joint (integer): index of the joint

            Returns:
            (double): position of the joint
        '''
        return self.q[joint]

    
    def getDOF(self):
        '''
        Method that returns number of DOF of the manipulator.
        '''
        return self.dof

class Task:
    '''
    Base class representing an abstract Task.
    '''
    def __init__(self, name, desired):
        '''
            Constructor.

            Arguments:
            name (string): title of the task
            desired (Numpy array): desired sigma (goal)
        '''
        self.name = name # task title
        self.sigma_d = desired # desired sigma
        
    
    def update(self, robot):
        '''
        Method updating the task variables (abstract).

        Arguments:
        robot (object of class Manipulator): reference to the manipulator
        '''
        pass

    
    def setDesired(self, value):
        ''' 
        Method setting the desired sigma.

        Arguments:
        value(Numpy array): value of the desired sigma (goal)
        '''
        self.sigma_d = value

    
    def getDesired(self):
        '''
        Method returning the desired sigma.
        '''
        return self.sigma_d

    
    def getJacobian(self):
        '''
        Method returning the task Jacobian.
        '''
        return np.round(self.J, decimals=2)

        
    def getError(self):
        '''
        Method returning the task error (tilde sigma).
        '''
        return self.err

    def setK(self, K):
        '''
        Method setting the velocity gain.
        '''
        self.K = K

    def getK(self):
        '''
        Method returning the velocity gain.
        '''
        return self.K
    
    def setFF(self, ff):
        '''
        Method setting the velocity gain.
        '''
        self.ff = ff

    def getFF(self):
        '''
        Method returning the velocity gain.
        '''
        return self.ff

    def isActive(self, ):
        return True
    def euler_to_quaternion(self):
        (yaw, pitch, roll) = (self.sigma_d[5], self.sigma_d[4], self.sigma_d[3])
        qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
        qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
        qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
        qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
        return [qx, qy, qz, qw]

class Position2D(Task):
    '''
    Subclass of Task, representing the 2D position task.
    '''
    def __init__(self, name, link, desired):
        super().__init__(name, desired)
        self.J =   np.zeros((2,5))  # Initialize with proper dimensions
        self.err = np.zeros((2,1))  # Initialize with proper dimensions
        self.K = np.eye(2)
        self.ff = np.zeros((2,1))
        self.link = link

    def update(self, robot):
        self.J =  robot.getLinkJacobian(self.link)[0:2,:].reshape(2,5)                              # Update task Jacobian
       
        self.err = self.sigma_d.reshape(2,1) - robot.getLinkTransform(self.link)[0:2,3].reshape(2,1)             # Update task error
        self.dxe = self.getFF() + self.getK()@self.getError()

class Orientation2D(Task):
    '''
    Subclass of Task, representing the 2D orientation task.
    '''
    def __init__(self, name, link, desired):
        
        super().__init__(name, desired)
        self.J = np.zeros([1,5])                  # Initialize with proper dimensions
        self.err = np.zeros((1,1))                # Initialize with proper dimensions
        self.K = np.eye(1)
        self.ff = np.zeros((1,1))
        self.link = link

    def update(self, robot):
        self.J = robot.getLinkJacobian(self.link)[5,:].reshape(1,5)   # Update task Jacobian
        T = robot.getLinkTransform(self.link)
        self.err =  (self.sigma_d - np.arctan2(T[1,0],T[0,0])).reshape(1,1)                       # Update task error
        self.dxe = self.getFF() + self.getK()@self.getError()


class Configuration2D(Task):
    '''
    Subclass of Task, representing the 2D configuration task.
    '''
    def __init__(self, name, link, desired):
        super().__init__(name, desired)
        self.J = np.zeros((6,5)) # Initialize with proper dimensions
        self.err = np.zeros((6,1)) # Initialize with proper dimensions
        self.K = np.eye(6)
        self.ff = np.zeros((6,1))
        self.link = link
    def signum(self, x):
        return 1 if x >= 0 else -1
    def update(self, robot):
        T = robot.getLinkTransform(self.link)
        qw_ = 0.5*np.sqrt(T[0,0]+T[1,1]+T[2,2]+1)
        
        qe = 0.5*np.array([[self.signum(T[2,1]-T[1,2]) * np.sqrt(round(T[0,0]-T[1,1]-T[2,2]+1, 3)),
                            self.signum(T[0,2]-T[2,0]) * np.sqrt(round(T[1,1]-T[0,0]-T[2,2]+1, 3)),
                            self.signum(T[1,0]-T[0,1]) * np.sqrt(round(T[2,2]-T[0,0]-T[1,1]+1, 3))
                        ]])

        # convert euler orientation angle to quaternion 
        qx_d, qy_d, qz_d,qw_d = self.euler_to_quaternion()

        qe_d = np.array([[qx_d, qy_d, qz_d]])

        epselon_err = qw_ *qe_d - qw_d*qe - np.cross(qe,qe_d)
        
        eta_err = self.sigma_d[0:3].reshape(3,1) - T[0:3,3].reshape(3,1)

        self.J = robot.getLinkJacobian(self.link).reshape(6,5)  # Update task Jacobian
        self.err = np.concatenate([eta_err, epselon_err.T])     # Update task error
        self.dxe = self.getFF() + self.getK()@self.getError()
    



class JointPosition(Task):
    ''' 
        Subclass of Task, representing the joint position task.
    '''
    def __init__(self, name, joint, desired):
        '''
        Arguments:
        name (string): custom name 
        q (int): joint to set
        desired (double): selected joint angle(rad)
        '''
        super().__init__(name, desired)
        self.joint = joint
        self.J = np.zeros((1,5)) # Initialize with proper dimensions
        self.err = np.zeros((1,1)) # Initialize with proper dimensions
        self.K = np.eye(1)
        self.ff = np.zeros((1,1))
        
    def update(self, robot):
        self.q = robot.getJointPos(self.joint)
        T = robot.getEETransform()
        self.J[0, self.joint] = 1 #np.array([[0, 0, 1]])
        self.err = self.sigma_d - robot.getJointPos(self.joint)
        self.dxe = self.getFF() + self.getK()@self.getError()


class Obstacle2D(Task):
    '''
    
    '''
    def __init__(self, name, desired, r):
        super().__init__(name, desired)
        self.err = np.zeros((2,1))  # Initialize with proper dimensions
        self.r = r
        self.active = False
        self.link = 3
        # self.K = np.eye(2)
        # self.ff = np.zeros((2,1))    
    
    def update(self, robot):
        self.J =  robot.getLinkJacobian(self.link)[0:2,:].reshape(2,3)                                        # Update task Jacobian
        eta = robot.getLinkTransform(self.link)[0:2,3].reshape(2,1) 
        
        if (not self.active) and np.linalg.norm(eta - self.sigma_d) <= self.r[0]: 
            self.active = True 
            self.dxe = (eta - self.sigma_d)/np.linalg.norm(eta - self.sigma_d)
        if self.active and np.linalg.norm(eta - self.sigma_d) >= self.r[1]:
            self.active = False 
    
    def isActive(self):
        return self.active

class JointLimit2D(Task):
    '''
    
    '''
    def __init__(self, name, desired, jlimit):
        super().__init__(name, desired)
        self.J = np.zeros((1,5)) # Initialize with proper dimensions
        self.jlimit = jlimit
        self.joint = desired
        self.active = False
        self.dxe = 0
    
    def update(self, robot):
        self.J[0, self.joint] = 1
        self.q = robot.getJointPos(self.joint)
        if self.dxe==0 and self.q >= self.jlimit[1]-0.05:
            self.dxe = -1
            self.active = True

        if self.dxe==0 and self.q <= self.jlimit[0]+0.05:
            self.dxe = 1
            self.active = True

        if self.dxe==-1 and self.q <= self.jlimit[1]-0.09:
            self.dxe = 0
            self.active = False

        if self.dxe==1 and self.q >= self.jlimit[0]+0.09:
            self.dxe = 0
            self.active = False
            
    def isActive(self):
        return self.active