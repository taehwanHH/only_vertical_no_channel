import torch
import gym
import time
import numpy as np
import mujoco_py
from scipy.spatial.transform import Rotation as R
from param import Hyper_Param

DEVICE = Hyper_Param['DEVICE']

class RoboticEnv:
    def __init__(self, model, max_time=3000):
        # Define the state space and action space
        self.num_sensor_output = 1
        self.num_robot = 4
        self.state_dim = self.num_sensor_output * self.num_robot
        self.action_dim = self.num_robot
        self.state_space = gym.spaces.Box(low=0, high=1500, shape=(self.state_dim,))
        self.action_space = gym.spaces.Box(low=0, high=0.7, shape=(self.action_dim,))

        # mujoco-py
        self.sim = mujoco_py.MjSim(model)
        self.viewer = mujoco_py.MjViewer(self.sim)

        # initialize
        self.max_time = max_time
        self.time_step = 0
        self.state = torch.tensor([0]*self.state_dim)
        self.reward = torch.tensor(0)
        self.done = False
        self.box_init_pos = self.sim.data.body_xpos[self.sim.model.body_name2id("box")]
        print(self.box_init_pos)

    def step(self, action):
        self.time_step += 1

        for i in range(self.num_robot):
            actuator_idx = self.sim.model.actuator_name2id(f"{i+1}_actuator_joint2")
            self.sim.data.ctrl[actuator_idx] = action[i]
        self.sim.step()
        time.sleep(0.01)
        touch_vector = []
        for i in range(self.num_robot):
            # sensor_idx = self.sim.model.sensor_adr[self.sim.model.sensor_name2id(f"touch_sensor{i+1}")]
            sensor_idx = self.sim.model.sensor_name2id(f"touch_sensor{i + 1}")
            # print(self.sim.data.sensordata[sensor_idx])
            touch_vector.append(self.sim.data.sensordata[sensor_idx])
        next_state = torch.tensor(touch_vector, dtype=torch.float32).to(DEVICE)

        self.viewer.render()
        # Get box orientation quaternion
        object_quat = self.sim.data.body_xquat[self.sim.model.body_name2id("box")]
        # Convert quaternion to Euler angles
        object_euler = torch.tensor(R.from_quat(object_quat).as_euler('xyz', degrees=True),device=DEVICE,dtype=torch.float32)
        reward = torch.square(object_euler[1]) + torch.square(object_euler[2])
        reward = reward.to(DEVICE)

        # Get box position
        box_pos = self.sim.data.body_xpos[self.sim.model.body_name2id("box")]
        box_z_pos = box_pos[2]

        # if self.time_step > self.max_time or box_z_pos < 0.2:
        print(next_state)
        if self.time_step > self.max_time or (next_state.sum()==0 and self.time_step > 100):
            self.done = True

        return next_state, reward, self.done, {}

    def reset(self):
        self.time_step = 0
        self.done = False

        past_box_init = self.box_init_pos
        std_dev = [0.1, 0.1, 0.1]
        new_box_pos = np.random.normal(past_box_init, std_dev)

        self.sim.reset()

        touch_vector = []
        for i in range(self.num_robot):
            sensor_idx = self.sim.model.sensor_adr[self.sim.model.sensor_name2id(f"touch_sensor{i + 1}")]
            touch_vector.append(self.sim.data.sensordata[sensor_idx])

        state = torch.tensor(touch_vector, device=DEVICE,dtype=torch.float32)

        return state











