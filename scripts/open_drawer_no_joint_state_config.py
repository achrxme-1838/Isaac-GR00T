"""OpenDrawer fine-tuning with the four arm joint state fields excluded.

Pass --modality-config-path scripts/open_drawer_no_joint_state_config.py to examples/finetune.sh.
Video, action and language settings follow the built-in PandaOmron configuration.
"""

from copy import deepcopy

from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS
from gr00t.data.types import ModalityConfig


config = deepcopy(MODALITY_CONFIGS["robocasa_panda_omron"])
config["state"] = ModalityConfig(
    delta_indices=[0],
    modality_keys=[
        "gripper_qpos",
        "base_position",
        "base_rotation",
        "end_effector_position_relative",
        "end_effector_rotation_relative",
        "gripper_qvel",
        "end_effector_position_absolute",
        "end_effector_rotation_absolute",
    ],
)

# Override the existing tag; register_modality_config only accepts new tags.
MODALITY_CONFIGS["robocasa_panda_omron"] = config
