import xml.etree.ElementTree as ET
import numpy as np

from robosuite.models import MujocoWorldBase
from robosuite.utils.mjcf_utils import new_actuator, new_joint
from robosuite.utils.binding_utils import MjSim, MjRenderContextOffscreen
from robosuite.utils import OpenCVRenderer
from robosuite.models.arenas.table_arena import TableArena
from robosuite.models.objects import BoxObject


def create_vertical_motion_robot(name, pos, rotation):
    base_body = ET.Element("body", name=name, pos=pos)
    base_body.set("quat", rotation)

    link1 = ET.SubElement(base_body, "body", name=f"{name}_link1", pos="0 0 0")
    vertical_joint = new_joint(name=f"{name}_vertical_joint", type="slide", axis="0 0 1", pos="0 0 0", range="0 2")
    link1.append(vertical_joint)
    ET.SubElement(link1, "geom", name=f"{name}_link1_geom", type="capsule", size="0.05 0.2", pos="0 0 0.1",rgba="1 1 0 1", contype="1", conaffinity="1")

    link2 = ET.SubElement(link1, "body", name=f"{name}_link2", pos="0 0 0.5")
    vertical_joint2 = new_joint(name=f"{name}_vertical_joint2", type="slide", axis="0 0 1", pos="0 0 0", range="0 2", armature = "0.01",stiffness="2000",damping="5")
    link2.append(vertical_joint2)
    ET.SubElement(link2, "geom", name=f"{name}_link2_geom", type="cylinder", size="0.05 0.4", pos="0 0 0.2 ",contype="1", conaffinity="1")


    guide = ET.SubElement(link1, "body", name=f"{name}_guide", pos="0 0 0")
    ET.SubElement(guide, "geom",name=f"{name}_guide_geom", type="cylinder", size="0.025 1", pos="0 0 0.5")

    # End effector with force sensor at the very end
    end_effector = ET.SubElement(link2, "body", name=f"{name}_end_effector", pos="0 0 0.5")
    ET.SubElement(end_effector, "geom", name=f"{name}_ee_geom", type="sphere", size="0.05", pos="0 0 0.1")

    return base_body

def add_force_sensor(world, body, site_name, sensor_name):
    # Define the site where the force sensor will be located
    sensor_site = ET.SubElement(body, "site", name=site_name, type="sphere", pos="0 0 0.1", size="0.05", rgba="0 0 1 1")
    # Define the force sensor
    force_sensor = ET.SubElement(world.sensor, "force", name=sensor_name, site=site_name)
    return sensor_site, force_sensor

world = MujocoWorldBase()

arena = TableArena(table_full_size=(1.0, 1.0, 0.05), table_offset=(-1, 0, 0.1), has_legs=False)
world.merge(arena)


robot_positions = ["-0.7 0.3 0.1", "-1.3 0.3 0.1", "-0.7 -0.3 0.1", "-1.3 -0.3 0.1"]
robot_rotations = ["-0.3827 0 0 0.9239", "-0.9239 0 0 0.3827", "0.3827 0 0 0.9239", "0.9239 0 0 0.3827"]
robots = []

for i, (pos, rot) in enumerate(zip(robot_positions, robot_rotations)):
    robot = create_vertical_motion_robot(f"robot{i+1}", pos, rot)
    world.worldbody.append(robot)
    robots.append(robot)

# Add force sensors to the end effectors
for i in range(1, 5):
    end_effector = world.worldbody.find(f".//body[@name='robot{i}_end_effector']")
    add_force_sensor(world, end_effector, f"ee_force_site{i}", f"force_sensor{i}")


for i in range(1, 5):
    world.actuator.append(new_actuator(joint=f"robot{i}_vertical_joint", act_type="position", name=f"{i}_actuator_joint", kv="500"))
    world.actuator.append(
        new_actuator(joint=f"robot{i}_vertical_joint2", act_type="position", name=f"{i}_actuator_joint2", kp="10000",kv='600'))

for i in range(1, 5):
    world.contact.append(ET.Element("pair",geom1=f"robot{i}_link1_geom",geom2=f"robot{i}_link2_geom"))

# Add a box object to the world
box = BoxObject(
    name="box", size=[0.6, 0.6, 0.1], rgba=[1, 0, 0, 1], friction=[1, 0.005, 0.0001]
).get_obj()
box.set("pos", "-1 0 2")
world.worldbody.append(box)


camera = ET.SubElement(world.worldbody, "camera", name="overview", pos="0 0 8", mode="fixed",  quat="0.3827 0  0 0.9239", fovy="45")

if __name__ == "__main__":
    model = world.get_model(mode="mujoco")

    sim = MjSim(model)
    render_context = MjRenderContextOffscreen(sim, device_id=-1)
    viewer = OpenCVRenderer(sim)
    sim.add_render_context(render_context)

    render_context.cam.type = 2
    render_context.cam.fixedcamid = sim.model.camera_name2id("overview")
    render_context.cam.lookat[:] = [0,0,0]
    render_context.cam.distance = 3
    render_context.cam.elevation = -30
    render_context.cam.azimuth = 100

    t = 0

    for _ in range(5000):
        pos = 4 * np.sin(t / 50)  # 주기를 줄여 움직임을 더 자주 확인

        force_values =[]
        for i in range(1, 5):
            current_position = sim.data.qpos[sim.model.joint_name2id(f"robot{i}_vertical_joint")]
            sim.model.geom_size[sim.model.geom_name2id(f"robot{i}_guide_geom")] = [0.025, 0.3 ,0]
            sim.data.ctrl[sim.model.actuator_name2id(f"{i}_actuator_joint")] = 0
            sim.data.ctrl[sim.model.actuator_name2id(f"{i}_actuator_joint2")] = abs( 0.15*i*np.sin(t/100))

            sensor_start_idx = sim.model.sensor_adr[sim.model.sensor_name2id(f"force_sensor{i}")]
            force_vector = sim.data.sensordata[sensor_start_idx:sensor_start_idx + 3]  # x, y, z components

            print(f"Robot{i} Force sensor values: {force_vector}")

        print("---------------------------------------")
        sim.step()
        viewer.render()

        t += 1
