from config_robot import world
import numpy as np
from robosuite.utils.binding_utils import MjSim, MjRenderContextOffscreen
from robosuite.utils import OpenCVRenderer
from scipy.spatial.transform import Rotation as R


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
        sim.data.ctrl[sim.model.actuator_name2id(f"{i}_actuator_joint")] = 0
        sim.data.ctrl[sim.model.actuator_name2id(f"{i}_actuator_joint2")] = abs(0.15 * i * np.sin(t / 100))
        # sim.data.ctrl[sim.model.actuator_name2id(f"{i}_actuator_joint2")] = 4

        sensor_idx = sim.model.sensor_adr[sim.model.sensor_name2id(f"touch_sensor{i}")]
        touch_vector = sim.data.sensordata[sensor_idx]  # x, y, z components

        # print(f"Robot{i} Force sensor values: {touch_vector}")
    # Get box orientation quaternion
    box_quat = sim.data.body_xquat[sim.model.body_name2id("box_main")]
    # Convert quaternion to Euler angles
    box_euler = R.from_quat(box_quat).as_euler('xyz', degrees=True)
    # print(f"Box orientation (Euler angles): {box_euler}")
    # # Get accelerometer data
    # acc_start_idx = sim.model.sensor_adr[sim.model.sensor_name2id("box_accelerometer")]
    # acc_values = sim.data.sensordata[acc_start_idx:acc_start_idx + 3]
    # print(f"Box accelerometer values: {acc_values}")
    box_pos = sim.data.body_xpos[sim.model.body_name2id("box_main")]
    box_z_pos = box_pos[2]
    print(f"Box z position: {box_z_pos}")

    print("---------------------------------------")
    sim.step()
    viewer.render()

    t += 1