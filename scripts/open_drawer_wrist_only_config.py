"""OpenDrawer fine-tuning with only the wrist camera and all 12 state fields.

Pass --modality-config-path scripts/open_drawer_wrist_only_config.py to examples/finetune.sh.
Action and language settings follow the built-in PandaOmron configuration.
"""

from copy import deepcopy

from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS
from gr00t.data.types import ModalityConfig


config = deepcopy(MODALITY_CONFIGS["robocasa_panda_omron"])
config["video"] = ModalityConfig(
    delta_indices=[0],
    modality_keys=["res256_image_wrist_0"],
    # Keep the simulator's key while reading the dataset's wrist camera.
    video_key_mapping={"res256_image_wrist_0": "wrist_view"},
)
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
        "joint_position",
        "joint_position_cos",
        "joint_position_sin",
        "joint_velocity",
    ],
)

# Override the existing tag; register_modality_config only accepts new tags.
MODALITY_CONFIGS["robocasa_panda_omron"] = config
