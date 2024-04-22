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

import matplotlib.pyplot as plt

ModelPath = './robomaster_wall.xml'
LIN_VEL_STEP_SIZE = 0.1
ANG_VEL_STEP_SIZE = 0.1

class KeyboardControl(object):
    def __init__(self):
        self.settings = termios.tcgetattr(sys.stdin)
        self.linear_vel=0.0
        self.angular_vel=0.0

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
        if key is not None:
            if key == 'q':    # Quit
                    self.linear_vel=0.0
                    self.angular_vel=0.0
            elif key == 'w':    # Forward
                    self.linear_vel += LIN_VEL_STEP_SIZE
            elif key == 's':    # Reverse
                    self.linear_vel -= LIN_VEL_STEP_SIZE
            elif key == 'd':    # Right
                    self.angular_vel -= ANG_VEL_STEP_SIZE
            elif key == 'a':    # Left
                    self.angular_vel += ANG_VEL_STEP_SIZE


            if self.angular_vel>1.0:
                 self.angular_vel=1.0
            if self.angular_vel<-1.0:
                 self.angular_vel=-1.0

            if self.linear_vel>1.0:
                 self.linear_vel=1.0
            if self.linear_vel<-1.0:
                 self.linear_vel=-1.0

        return (self.linear_vel, self.angular_vel)

    def vel_controller(self, m, d, vels):
        d.actuator('forward').ctrl[0] = vels[0]
        d.actuator('turn').ctrl[0] = vels[1]
    

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

    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot()
    acc_data = []
    with mujoco.viewer.launch_passive(m, d) as viewer:
        start = time.time()
        i = 0
        while viewer.is_running(): #and time.time() - start < 30:
            step_start = time.time()
            if i%10==0:
                vels = node.run_keyboard_control()
                print(vels)
            i+=1
            mujoco.set_mjcb_control(lambda m, d: node.vel_controller(m, d, vels))

            mujoco.mj_step(m, d)

            acc_data.append(d.sensor("imu").data.copy())

            renderer.update_scene(d, camera="fixater")
            cam_img = renderer.render()

            
            # with viewer.lock():
            #     viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(d.time % 2)
            
            viewer.sync()

            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
    acc_data = np.array(acc_data)
    plt.plot(acc_data)
    plt.show()
    plt.imshow(cam_img)
    im = Image.fromarray(cam_img)
    im.save("test_img.png")
    plt.show(block=True)