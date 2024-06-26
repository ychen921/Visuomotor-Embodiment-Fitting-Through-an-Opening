import mujoco
import mujoco.viewer as viewer

import numpy as np
import time
import sys
import select
import tty
import termios
from PIL import Image
from pynput import keyboard
from multiprocessing import Process
import matplotlib.pyplot as plt
import cv2

from util import PhiConstraintSolver, find_corners, find_phi, camera_matrix, compute_3d, ball_detection

ModelPath = './model/robomaster_wide_opening.xml'
LIN_VEL_STEP_SIZE = 0.1
ANG_VEL_STEP_SIZE = 0.1
RES_X = 640
RES_Y = 480


class KeyboardControl(object):
    def __init__(self):
        self.settings = termios.tcgetattr(sys.stdin)
        self.forward_vel = 0.0
        self.lateral_vel = 0.0
        self.angular_vel = 0.0
        self.stop = False

    def getKey(self):
        """Get the key that is pressed"""
        tty.setraw(sys.stdin.fileno())
        rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
        if rlist:
            key = sys.stdin.read(1)
        else:
            key = ''

        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.settings)
        return key

    def run_keyboard_control(self):
        key = self.getKey()
        if not key:
            return (self.forward_vel, self.lateral_vel, self.angular_vel)
        if key == 'q':    # Quit
            self.forward_vel = 0.0
            self.lateral_vel = 0.0
            self.angular_vel = 0.0
        elif key == 'w':    # Forward
            self.forward_vel += LIN_VEL_STEP_SIZE
        elif key == 's':    # Reverse
            self.forward_vel -= LIN_VEL_STEP_SIZE
        elif key == 'd':    # Right
            self.lateral_vel += ANG_VEL_STEP_SIZE
        elif key == 'a':    # Left
            self.lateral_vel -= ANG_VEL_STEP_SIZE

        if self.angular_vel > 1.0:
            self.angular_vel = 1.0
        if self.angular_vel < -1.0:
            self.angular_vel = -1.0

        if self.forward_vel > 1.0:
            self.forward_vel = 1.0
        if self.forward_vel < -1.0:
            self.forward_vel = -1.0

        if self.lateral_vel > 1.0:
            self.lateral_vel = 1.0
        if self.lateral_vel < -1.0:
            self.lateral_vel = -1.0

        return (self.forward_vel, self.lateral_vel, self.angular_vel)

    def vel_controller(self, m, d, vels):
        d.actuator('forward').ctrl[0] = vels[0]
        d.actuator("horizontal").ctrl[0] = vels[1]
        d.actuator('turn').ctrl[0] = vels[2]


def load_callback(m=None, d=None):
    mujoco.set_mjcb_control(None)
    m = mujoco.MjModel.from_xml_path(ModelPath)
    d = mujoco.MjData(m)

    if m is not None:
        vels = node.run_keyboard_control()
        print(vels)
        mujoco.set_mjcb_control(lambda m, d: node.vel_controller(m, d, vels))

    return m, d


if __name__ == '__main__':
    node = KeyboardControl()

    m = mujoco.MjModel.from_xml_path(ModelPath)
    d = mujoco.MjData(m)

    # Make all the things needed to render a simulated camera
    gl_ctx = mujoco.GLContext(RES_X, RES_Y)
    gl_ctx.make_current()

    scn = mujoco.MjvScene(m, maxgeom=100)

    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
    cam.fixedcamid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, 'fixater')
    cam_fovy = m.cam_fovy[cam.fixedcamid]

    # get focal distance & camera matrix
    f, K = camera_matrix(fovy=cam_fovy, height=RES_Y, width=RES_X)

    vopt = mujoco.MjvOption()
    pert = mujoco.MjvPerturb()

    ctx = mujoco.MjrContext(m, mujoco.mjtFontScale.mjFONTSCALE_150)
    mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, ctx)

    viewport = mujoco.MjrRect(0, 0, RES_X, RES_Y)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot()
    acc_data = []

    last_frame_ts = 0.0
    frame_count = 0
    frame_rate = 30.0 # 30 fps
    frame_period = 1.0/frame_rate

    command_period = 2.0
    vels = [0, 0, 0]
    mujoco.set_mjcb_control(lambda m, d: node.vel_controller(m, d, vels))

    # create a solver for each of the 4 corners
    solvers = [PhiConstraintSolver(dt=frame_period) for _ in range(4)]
    start_time = 3.0
    switch_time = 2
    iswitch = 1
    first_switch = True
    corners_0 = None
    started = False

    hor_switch = 10
    hor_flag = False
    forward_switch = 35
    forward_flag = False

    a = 1
    b = 3
    w = 2*np.pi/4.0
    vx = 1.5
    vy = 6.5
    v_scale = 0.0
    Z0s_all = []

    Kp = 0.1
    Kd = 0.001
    e_hor = None
    e_hor_last = None
    e_hor_diff = None
    e_hor_sm = None
    with mujoco.viewer.launch_passive(m, d) as viewer:
        start = time.time()
        i = 0
        while viewer.is_running():  # and time.time() - start < 30:
            step_start = time.time()
            
            if d.time > start_time and d.time < hor_switch:
                started = True
            
                t = d.time-start_time
                vx = 2*w*a*np.cos(w*t-np.pi/2)
                vy = -2*w*b*np.sin(w*t-np.pi/2)

                vels[0] = vx
                vels[1] = vy

            # # Move horizonlly
            if d.time > hor_switch and hor_flag is False:
                vels[0] = 0
                hor_flag = True

            # Move forward to the wall
            # if d.time > forward_switch:
            #     hor_flag = False
            #     forward_flag = True
            #     vels[0] = 8.0
            #     vels[1] = 0.0

            mujoco.mj_step(m, d)
            acc_data = d.sensor('imu').data.copy()  # ndarray
            contact = d.ncon
            
            # if forward_flag == True and contact > 4:
            #     # print('Collision:', contact)
            #     vels[0] = 0.0
            #     vels[1] = 0.0

            if contact > 4:
                print("cannot pass through opening")
                print('Collision:', contact)
                vels[0] = 0.0
                vels[1] = 0.0
                break


            if d.time >= frame_period*frame_count:
                frame_count += 1
                
                # Render the simulated camera
                mujoco.mjv_updateScene(m, d, vopt, pert, cam, mujoco.mjtCatBit.mjCAT_ALL, scn)
                mujoco.mjr_render(viewport, scn, ctx)
                cam_img = np.empty((RES_Y, RES_X, 3), dtype=np.uint8)
                mujoco.mjr_readPixels(cam_img, None, viewport, ctx)

                # OpenGL renders with inverted y axis
                cam_img = cv2.flip(cam_img, 0)

                # Show the simulated camera image
                cv2.imshow('fixation', cv2.cvtColor(cam_img, cv2.COLOR_RGB2BGR))
                cv2.waitKey(1)

                # if forward_flag == True and contact > 4:
                #     ball_centers, distance = ball_detection(cam_img) # ball centers
                    
                #     print('\n#========= Site centers =========#')
                #     print('Collisions: ', contact)
                #     print('Centers: ', ball_centers)
                #     print('Distance: ', distance)
                #     print('#===============================#')
                #     continue

                if forward_flag == True:
                    continue

                # process the corners (x, y)
                corners = find_corners(cam_img)

                if hor_flag == True:
                    if e_hor is None:
                        e_hor = np.mean(corners[:,0])-RES_X/2
                        e_hor_last = e_hor
                        e_hor_sm = e_hor
                        vy = Kp*e_hor
                    else:
                        e_hor = np.mean(corners[:,0])-RES_X/2
                        e_hor_sm = e_hor_sm*0.9 + e_hor*0.1
                        e_hor_diff = e_hor-e_hor_last
                        e_hor_last = e_hor
                        vy = Kp*e_hor + Kd*e_hor_diff
                    print(f"ERROR : {e_hor}, {e_hor_sm}")
                    if np.abs(e_hor_sm) < 5:
                        forward_flag = True
                        vels[0] = 8.0
                        vels[1] = 0.0
                        continue
                    else:
                        vels[1] = -vy

                if corners_0 is None:
                    corners_0 = corners
                    
                phi_matrices = find_phi(corners_0=corners_0,
                                        corners_t=corners)
                
                for i in range(4):
                    solvers[i].accumulate(acc=acc_data, phi=phi_matrices[i],
                                          curr_t=d.time,z_only=True)
                    
            # with viewer.lock():
            #     viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(d.time % 2)

            viewer.sync()
            
            if started and forward_flag == False:
                Z0s = []
                for i in range(4):
                    ans = solvers[i].solve()
                    Z0s_all.append(-1*ans[0])
                    Z0s.append(-1*ans[0])
                    # print(ans)
                    
                # Compute X0, Y0 by Z0, corners coordinates, focal length
                pts0_3d = compute_3d(corners_0=corners_0, Z0s=Z0s, fl=f)
                print("================")
                print(pts0_3d)

    acc_data = np.array(acc_data)

    plt.plot(solvers[0].acc_history)
    plt.show(block=True)

    plt.plot(Z0s_all)
    plt.show()

    plt.scatter(pts0_3d[:,0],pts0_3d[:,1])
    plt.gca().set_aspect("equal",adjustable="box")
    plt.show()
