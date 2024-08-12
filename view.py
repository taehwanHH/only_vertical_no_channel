import os
import mujoco_py
import time
import numpy as np
# import glfw

# XML 파일 경로
xml_path = "sim_env.xml"

# 모델 로드
model = mujoco_py.load_model_from_path(xml_path)
sim = mujoco_py.MjSim(model)

# MuJoCo 뷰어 생성
viewer = mujoco_py.MjViewer(sim)

t = 0
while True:
    if t % 30 == 0:
        for i in range(1, 5):
            sim.data.ctrl[sim.model.actuator_name2id(f"{i}_actuator_joint2")] = np.random.uniform(low=0, high=0.7)
            sim.data.ctrl[sim.model.actuator_name2id(f"{i}_actuator_joint3")] = np.random.uniform(low=0, high=0.7)


        # sim.data.ctrl[sim.model.actuator_name2id("1_actuator_joint2")] = np.random.uniform(low=0, high=0.4)
        # sim.data.ctrl[sim.model.actuator_name2id("1_actuator_joint3")] = np.random.uniform(low=0, high=0.4)
        # sim.data.ctrl[sim.model.actuator_name2id("1_actuator_joint2")] = 0
        # sim.data.ctrl[sim.model.actuator_name2id("1_actuator_joint3")] = 0

        # for i in range(1, 5):
        #     # sim.data.ctrl[sim.model.actuator_name2id(f"{i}_actuator_joint")] = np.random.uniform(low=0, high=0.2)
        #     sim.data.ctrl[sim.model.actuator_name2id(f"{i}_actuator_joint2")] = 0
        #     sim.data.ctrl[sim.model.actuator_name2id(f"{i}_actuator_joint")] = 0
        #     # sim.data.ctrl[sim.model.actuator_name2id(f"{i}_actuator_joint2")] = 0

    sim.step()
    t += 1
    # 렌더링
    viewer.render()

    # 잠시 대기
    time.sleep(0.01)  # 10ms 대기

