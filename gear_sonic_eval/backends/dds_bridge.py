"""Shared helper for constructing gear_sonic's DDS bridge safely.

``UnitreeSdk2Bridge.__init__`` starts the DDS reader threads
(``ChannelSubscriber.Init(handler, ...)``) *before* it creates the mutexes those
handlers take:

    self.left_hand_cmd_suber.Init(self.LeftHandCmdHandler, 1)   # thread starts
    ...
    self.left_hand_cmd_lock = threading.Lock()                  # created later

If a hand command arrives inside that window the reader thread dies with
``AttributeError: 'UnitreeSdk2Bridge' object has no attribute
'left_hand_cmd_lock'``.  It is a race in gear_sonic, not in this benchmark, and
it is likelier here because the deploy binary is already running and publishing
when the bridge is created.

Rather than patch gear_sonic, install the locks as *class* attributes first, so
a handler that runs early finds a valid lock. The instance attributes created a
few lines later simply shadow them.
"""

from __future__ import annotations

import threading


def create_bridge(wbc_config):
    """Construct UnitreeSdk2Bridge without the handler/lock start-up race."""
    from gear_sonic.utils.mujoco_sim.unitree_sdk2py_bridge import UnitreeSdk2Bridge

    for name in ("low_cmd_lock", "left_hand_cmd_lock", "right_hand_cmd_lock"):
        if not hasattr(UnitreeSdk2Bridge, name):
            setattr(UnitreeSdk2Bridge, name, threading.Lock())
    bridge = UnitreeSdk2Bridge(wbc_config)
    if wbc_config.get("USE_JOYSTICK"):
        bridge.SetupJoystick(
            device_id=wbc_config["JOYSTICK_DEVICE"], js_type=wbc_config["JOYSTICK_TYPE"]
        )
    return bridge
