#!/usr/bin/env python3
"""Re-extract the physical constants in ``configs/g1_physics_reference.json``.

Prints the MuJoCo and URDF total masses, the MuJoCo geom friction and both
models' foot collision geometry, so the numbers the IsaacLab backend uses to
close the sim-to-sim gap can be re-derived whenever a model changes.

    python tools/dump_physics_reference.py
"""

from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
MJ_ROBOT = ROOT / "gear_sonic/data/robot_model/model_data/g1/g1_29dof_with_hand.xml"
MJ_SCENE = ROOT / "gear_sonic/data/robot_model/model_data/g1/scene_43dof.xml"
URDF = ROOT / "gear_sonic/data/assets/robot_description/urdf/g1/main.urdf"


def mujoco_mass() -> float:
    root = ET.parse(MJ_ROBOT).getroot()
    return sum(float(b.find("inertial").get("mass"))
               for b in root.iter("body") if b.find("inertial") is not None)


def urdf_mass() -> float:
    root = ET.parse(URDF).getroot()
    return sum(float(l.find("inertial").find("mass").get("value"))
               for l in root.iter("link") if l.find("inertial") is not None)


if __name__ == "__main__":
    print(f"mujoco total mass : {mujoco_mass():.4f} kg")
    print(f"urdf   total mass : {urdf_mass():.4f} kg")
    scene = ET.parse(MJ_SCENE).getroot()
    for d in scene.iter("default"):
        for g in d.findall("geom"):
            print(f"mujoco geom friction: {g.get('friction')}")
    robot = ET.parse(MJ_ROBOT).getroot()
    for b in robot.iter("body"):
        if b.get("name") == "left_ankle_roll_link":
            for g in b.findall("geom"):
                if g.get("type") != "mesh":
                    print(f"mujoco foot collision: type={g.get('type')} size={g.get('size')} pos={g.get('pos')}")
    urdf = ET.parse(URDF).getroot()
    for l in urdf.iter("link"):
        if l.get("name") == "left_ankle_roll_link":
            shapes = [c.find("geometry")[0].tag for c in l.findall("collision")]
            print(f"urdf foot collision: {len(shapes)} x {set(shapes)}")
