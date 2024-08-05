from config_robot import world
import numpy as np
from robosuite.utils.binding_utils import MjSim, MjRenderContextOffscreen
from robosuite.utils import OpenCVRenderer


model = world.get_model(mode="mujoco")

sim = MjSim(model)
render_context = MjRenderContextOffscreen(sim, device_id=-1)
viewer = OpenCVRenderer(sim)
sim.add_render_context(render_context)

render_context.cam.type = 2
render_context.cam.fixedcamid = sim.model.camera_name2id("overview")

t = 0

for _ in range(5000):
    pos = 4 * np.sin(t / 50)  # 주기를 줄여 움직임을 더 자주 확인

    force_values = []
    for i in range(1, 5):
        current_position = sim.data.qpos[sim.model.joint_name2id(f"robot{i}_vertical_joint")]
        sim.model.geom_size[sim.model.geom_name2id(f"robot{i}_guide_geom")] = [0.025, 0.3, 0]
        sim.data.ctrl[sim.model.actuator_name2id(f"{i}_actuator_joint")] = 0
        sim.data.ctrl[sim.model.actuator_name2id(f"{i}_actuator_joint2")] = abs(0.15 * i * np.sin(t / 100))

        sensor_start_idx = sim.model.sensor_adr[sim.model.sensor_name2id(f"force_sensor{i}")]
        force_vector = sim.data.sensordata[sensor_start_idx:sensor_start_idx + 3]  # x, y, z components

        print(f"Robot{i} Force sensor values: {force_vector}")

    print("---------------------------------------")
    sim.step()
    viewer.render()

    t += 1