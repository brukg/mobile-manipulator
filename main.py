from mobile_manipulator import * # Includes numpy import
import matplotlib.pyplot as plt
import matplotlib.patches as patch
import matplotlib.animation as anim
import matplotlib.transforms as trans

# Robot model
d = np.zeros(3)                             # displacement along Z-axis
q = np.array([0.2, 0.3, 0.5])               # rotation around Z-axis (theta)
alpha = np.zeros(3)                         # displacement along X-axis
a = np.array([0.75, 0.5, 0.3])              # rotation around X-axis 
revolute = [True, True, True]               # flags specifying the type of joints

robot = MobileManipulator(d, q, a, alpha, revolute) # Manipulator object


# Task definition
base_pos = np.array([0.0, 0.0, 0, 0, 0, 0.0])
ee_pos = np.array([1.0, 0.5])

joint = 2 # select joint [0-5]
joint_angle = 0.3 # selected joint angle [-pi, pi]

joint_max = 0.1
joint_min = -0.5
tasks = [ 
            # Position2D("base position", 5, ee_pos[0:2].reshape(2,1)),
            # Orientation2D("base position", 3, 1),
            # Configuration2D("base position", 4, base_pos),
            JointLimit2D("Joint 1 limit",     joint, np.array([joint_min, joint_max])),
            Position2D("base position", 5, ee_pos),
            # Configuration2D("End-effector position", 5, ee_pos),
            # JointPosition("Joint position", joint, joint_angle),
        ] 

# Simulation params
dt = 1.0/60.0


def click(event): #get position from mouse
    global x, y, ee_pos
    ee_pos[0] = event.xdata
    ee_pos[1] = event.ydata

# Drawing preparation
fig = plt.figure()
fig.canvas.mpl_connect("button_press_event", click)
ax1 = fig.add_subplot(121, autoscale_on=False, xlim=(-2, 2), ylim=(-2,2))
ax1.set_title('Simulation')
ax1.set_aspect('equal')
ax1.grid()
ax1.set_xlabel('x[m]')
ax1.set_ylabel('y[m]')
rectangle = patch.Rectangle((-0.25, -0.15), 0.5, 0.3, color='blue', alpha=0.3)
veh = ax1.add_patch(rectangle)
line, = ax1.plot([], [], 'o-', lw=2) # Robot structure
path, = ax1.plot([], [], 'c-', lw=1) # End-effector path
point, = ax1.plot([], [], 'rx') # Target
PPx = []
PPy = []
ax1.legend(loc="upper left")

ax2 = fig.add_subplot(122, xlim=(0,40), ylim=(-2,2))
ax2.set_xlabel('t[s]')
ax2.set_ylabel('error')
ax2.grid()
lowerlimit = ax2.axhline(y = joint_min, color = 'r', linestyle = '--')
uperlimit = ax2.axhline(y = joint_max, color = 'r', linestyle = '--')

j1, = ax2.plot([], [], color='blue', lw=1, label="j_1(joint 1 position)") # joint 1
e1, = ax2.plot([], [], color='orange', lw=1, label="e_1(EE position error )") # End-effector path
ax2.legend(loc="upper left")
t_mem = [0]
j1_mem = [0]
e1_mem = [0]



# Simulation initialization
def init():
    global tasks, ee_pos
    line.set_data([], [])
    path.set_data([], [])
    point.set_data([], [])
    j1.set_data([], [])
    e1.set_data([], [])

    # random postion between [-2,2]
    tasks[1].sigma_d = np.random.choice([-2,2])*np.random.rand(2)

    return line, path, point, j1, e1

# Simulation loop
def simulate(t):
    global tasks
    global robot
    global PPx, PPy
    
    ### Recursive Task-Priority algorithm
    P = np.eye(5)
    dq = np.zeros([5,1])
    for i, t in enumerate(tasks):
       
        t.update(robot)
        J_bar = t.getJacobian()@P

        # W = 0.2*np.diag([2,3,4,1,4])

        J_bar_inv = DLS(J_bar, 0.05)
        if t.isActive():
            dq += J_bar_inv@(t.dxe - t.getJacobian()@dq)
            P  -= np.linalg.pinv(J_bar)@J_bar
           
                        

    # Update robot
    robot.update(dq, dt)
    
    # Update drawing
    # -- Manipulator links
    PP = robot.drawing()
    line.set_data(PP[0,:], PP[1,:])
    PPx.append(PP[0,-1])
    PPy.append(PP[1,-1])
    path.set_data(PPx, PPy)
    point.set_data(tasks[1].getDesired()[0], tasks[1].getDesired()[1])
    # -- Mobile base
    eta = robot.getBasePose()
    veh.set_transform(trans.Affine2D().rotate(eta[2,0]) + trans.Affine2D().translate(eta[0,0], eta[1,0]) + ax1.transData)


    t_mem.append(dt +t_mem[-1])
    e1_mem.append(np.linalg.norm(tasks[1].getError()))
    j1_mem.append(tasks[0].q[0])
    
    # EE position error and joint 1 position
    j1.set_data(t_mem, j1_mem)
    e1.set_data(t_mem, e1_mem)
    
    return line, veh, path, point, j1, e1

# Run simulation
animation = anim.FuncAnimation(fig, simulate, np.arange(0, 5, dt), 
                                interval=10, blit=True, init_func=init, repeat=True)
plt.show()