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

from util import PhiConstraintSolver, find_corners, find_phi

ModelPath = './robomaster_wall_v2.xml'
LIN_VEL_STEP_SIZE = 0.1
ANG_VEL_STEP_SIZE = 0.1


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
    renderer = mujoco.Renderer(m, 480, 640)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot()
    acc_data = []

    last_frame_ts = 0.0
    frame_count = 0
    frame_rate = 30.0
    frame_period = 1.0/frame_rate

    command_period = 2.0
    vels = (0, 0, 0)
    mujoco.set_mjcb_control(lambda m, d: node.vel_controller(m, d, vels))

    # create a solver for each of the 4 corners
    solvers = [PhiConstraintSolver(dt=frame_period) for _ in range(4)]
    start_time = 3.0
    switch_time = 2.5
    iswitch = 1
    first_switch = True
    corners_0 = None
    started = False
    started_prev = False
    with mujoco.viewer.launch_passive(m, d) as viewer:
        start = time.time()
        i = 0
        while viewer.is_running():  # and time.time() - start < 30:
            step_start = time.time()
            # if i%100==0:
            #     vels = node.run_keyboard_control()
            #     print(vels)
            # i+=1
            if d.time > start_time and vels[1] == 0:
                vels = (0, 3.0, 0)
                started = True
            if first_switch and (d.time-start_time) > switch_time:
                first_switch = False
                vels = (0, -1*vels[1], 0)
                start_time = d.time
                switch_time *= 2
            else:
                if (d.time-start_time) > iswitch*switch_time:
                    vels = (0, -1*vels[1], 0)
                    iswitch += 1
            # mujoco.set_mjcb_control(lambda m, d: node.vel_controller(m, d, vels))

            mujoco.mj_step(m, d)
            acc_data = d.sensor('imu').data.copy()  # ndarray
            if d.time >= frame_period*frame_count:
                frame_count += 1
                renderer.update_scene(d, camera="fixater")
                cam_img = renderer.render()
                cv2.imshow('fixation', cv2.cvtColor(cam_img, cv2.COLOR_RGB2BGR))
                cv2.waitKey(1)

                # process the corners
                corners = find_corners(cam_img)

                if corners_0 is None:
                    corners_0 = corners
                    
                if started_prev:
                    phi_matrices = find_phi(corners_0=corners_0,
                                            corners_t=corners)
                    Z0s = []
                    for i in range(4):
                        solvers[i].accumulate(acc=acc_data, phi=phi_matrices[i],
                                              curr_t=d.time-start_time)
                        ans = solvers[i].solve()
                        Z0s.append(ans[0])
                
                    print("================")
                    print(Z0s)

            # with viewer.lock():
            #     viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(d.time % 2)

            viewer.sync()
            started_prev = started

    acc_data = np.array(acc_data)
    plt.imshow(cam_img)
    im = Image.fromarray(cam_img)
    im.save("test_img.png")
    plt.show(block=True)

    plt.plot(solvers[0].acc_history)
    plt.show()
