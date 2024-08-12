import xml.etree.ElementTree as ET
from robosuite.models import MujocoWorldBase
from robosuite.utils.mjcf_utils import new_actuator, new_joint
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
    ET.SubElement(link2, "geom", name=f"{name}_link2_geom", type="cylinder", size="0.05 0.3", pos="0 0 0.2 ",contype="1", conaffinity="1")

    guide = ET.SubElement(link1, "body", name=f"{name}_guide", pos="0 0 0")
    ET.SubElement(guide, "geom",name=f"{name}_guide_geom", type="cylinder", size="0.025 0.5", pos="0 0 0.5")

    # End effector with force sensor at the very end
    end_effector = ET.SubElement(link2, "body", name=f"{name}_end_effector", pos="0 0 0.4")
    ET.SubElement(end_effector, "geom", name=f"{name}_ee_geom", type="sphere", size="0.05", pos="0 0 0.1")

    return base_body


def add_touch_sensor(world, body, site_name, sensor_name):
    # Define the site where the force sensor will be located
    sensor_site = ET.SubElement(body, "site", name=site_name, type="sphere", pos="0 0 0.1", size="0.05", rgba="0 0 1 1")
    # Define the force sensor
    touch_sensor = ET.SubElement(world.sensor, "touch", name=sensor_name, site=site_name)
    return sensor_site, touch_sensor


def add_accelerometer_sensor(world, body, sensor_name):
    accelerometer = ET.SubElement(world.sensor, "accelerometer", name=sensor_name, site=body.find('site').get('name'))
    return accelerometer


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
    add_touch_sensor(world, end_effector, f"ee_touch_site{i}", f"touch_sensor{i}")
    world.actuator.append(new_actuator(joint=f"robot{i}_vertical_joint", act_type="position", name=f"{i}_actuator_joint", kv="500"))
    world.actuator.append(
        new_actuator(joint=f"robot{i}_vertical_joint2", act_type="position", name=f"{i}_actuator_joint2", kp="10000",kv='600'))
    world.contact.append(ET.Element("pair", geom1=f"robot{i}_link1_geom", geom2=f"robot{i}_link2_geom"))

# Add a box object to the world
box = BoxObject(
    name="box", size=[0.6, 0.6, 0.1], rgba=[1, 0, 0, 1], friction=[0.001, 0.005, 0.0001]
).get_obj()
box.set("pos", "-1 0 2")
# Add accelerometer sensor to the box


world.worldbody.append(box)
add_accelerometer_sensor(world, box, "box_accelerometer")

camera = ET.SubElement(world.worldbody, "camera", name="overview", pos="0 0 8", mode="fixed",  quat="0.3827 0  0 0.9239", fovy="45")
