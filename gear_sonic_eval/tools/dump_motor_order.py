#!/usr/bin/env python3
"""Regenerate ``configs/g1_motor_order.json`` from the MuJoCo robot model.

The DDS motor index (``LowState.motor_state[i]`` / ``LowCmd.motor_cmd[i]``) is
defined by the joint document order of the MuJoCo XML filtered exactly as
``DefaultEnv.init_scene`` filters it.  Any backend that speaks to the deploy
binary must use this order, so it is extracted from the model rather than
hard-coded.

    python tools/dump_motor_order.py [path/to/g1_29dof_with_hand.xml]
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

BODY_PARTS = ["hip", "knee", "ankle", "waist", "shoulder", "elbow", "wrist"]
DEFAULT_XML = (
    Path(__file__).resolve().parents[2]
    / "gear_sonic/data/robot_model/model_data/g1/g1_29dof_with_hand.xml"
)


def dump(xml_path: Path) -> dict:
    root = ET.parse(xml_path).getroot()
    body, left, right = [], [], []
    for joint in root.iter("joint"):
        name = joint.get("name")
        if not name:
            continue
        if any(part in name for part in BODY_PARTS):
            body.append(name)
        elif "left_hand" in name:
            left.append(name)
        elif "right_hand" in name:
            right.append(name)
    if len(body) != 29:
        raise SystemExit(f"expected 29 body joints, found {len(body)} in {xml_path}")
    return {"body_joints": body, "left_hand_joints": left, "right_hand_joints": right}


if __name__ == "__main__":
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_XML
    print(json.dumps(dump(path), indent=2))
