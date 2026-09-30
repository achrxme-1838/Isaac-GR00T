"""Evaluate the pretrained REAL_G1 policy on RoboCasa tabletop tasks.

This is the executable rollout client. Model-free regression tests for its
adapter live in ``tests/gr00t/experiment/test_real_g1_robocasa.py``.

Run the model server in the GR00T environment::

    python gr00t/eval/run_gr00t_server.py \
        --model-path nvidia/GR00T-N1.7-3B --embodiment-tag REAL_G1 \
        --use-sim-policy-wrapper

Run this client in the RoboCasa GR1 tabletop environment (see
``gr00t/eval/sim/robocasa-gr1-tabletop-tasks/setup_RoboCasaGR1TabletopTasks.sh``)::

    python gr00t/experiment/custom/run_gr00t_on_robocas.py \
        --wbc-path ../GR00T-WholeBodyControl --task PnPAppleToPlate

Alternatively, add this experiment's simulator dependencies to an existing
GR00T environment (which already supplies headless OpenCV and inference deps)::

    uv pip install --python "$CONDA_PREFIX/bin/python" --no-deps robosuite==1.5.1
    uv pip install --python "$CONDA_PREFIX/bin/python" \
        mujoco==3.2.6 numba==0.61.2 numpy==1.26.4 h5py lxml termcolor
    python external_dependencies/robocasa-gr1-tabletop-tasks/robocasa/scripts/download_tabletop_assets.py -y

The WBC checkout supplies the actual G1 robot, three-finger hands, meshes, and
joint ordering; the GR1 tabletop checkout supplies the tasks, not the robot.
Install the tabletop assets using that environment's setup script first.
Task distractors are disabled by default for this tabletop experiment. Use
``--use-distractors`` to enable them; some task-specific distractors (such as
PnPAppleToPlate's coffee_pod) are absent from the published asset archives.
The interactive MuJoCo viewer opens by default; drag/scroll to move the camera.
Use ``--no-visualize`` for headless execution. Video recording is opt-in via
``--video-dir outputs/real_g1_robocasa``.
``--smoke-test --max-episode-steps 20`` holds the initial joint positions to
check the simulator, observation history, action mapping, and viewer without a
model server. It does not measure policy performance.

This experiment fixes the lower body. It executes the model's decoded absolute
arm/hand/waist joint targets, not its redundant wrist EEF targets or navigation
and base-height commands. Relative-to-absolute conversion is already performed
by Gr00tPolicy: do not add the current joint positions again here.
The initial arm pose lifts the hands clear of the tabletop and is held during
object settling. The base is placed 0.30 m from the fixture's front reference
edge by default; --robot-table-distance adjusts this clearance.
"""

from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from typing import Literal
import xml.etree.ElementTree as ET

import gymnasium as gym
import numpy as np
import tyro


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gr00t.data.types import ModalityConfig  # noqa: E402
from gr00t.eval._horizon_contract import PolicyHorizonSpec  # noqa: E402
from gr00t.eval.sim.wrapper.multistep_wrapper import MultiStepWrapper  # noqa: E402
from gr00t.eval.sim.wrapper.video_recording_wrapper import VideoRecordingWrapper  # noqa: E402


JOINT_DIMS = {"left_arm": 7, "right_arm": 7, "left_hand": 7, "right_hand": 7, "waist": 3}
STATE_DIMS = {"left_wrist_eef_9d": 9, "right_wrist_eef_9d": 9, **JOINT_DIMS}
ACTION_DIMS = {**STATE_DIMS, "base_height_command": 1, "navigate_command": 3}
LANGUAGE_KEY = "annotation.human.task_description"


@dataclass
class Config:
    task: str = "PnPAppleToPlate"
    """RoboCasa tabletop task class name (not a GR1 gym environment ID)."""
    layout_id: int | None = None
    """Optional layout override; otherwise use the task's supported layouts."""
    style_id: int | None = None
    use_distractors: bool = False
    """Enable the task's extra distractors; requires all of their assets."""
    robot_base_height: float = 0.793
    """Fixed pelvis height in metres, matching WBC's G1 scene placement."""
    robot_table_distance: float = 0.30
    """Distance in metres from the fixed base to the fixture's front reference edge."""
    wbc_path: Path = REPO_ROOT.parent / "GR00T-WholeBodyControl"
    """Checkout containing decoupled_wbc's G1 robot and assets."""
    policy_host: str = "127.0.0.1"
    policy_port: int = 5555
    policy_timeout_ms: int = 120_000
    n_episodes: int = 1
    max_episode_steps: int = 400
    n_action_steps: int = 8
    seed: int = 0
    instruction: str | None = None
    """Override the task's language instruction, if set."""
    camera_name: Literal["robot0_oak_egoview", "robot0_rs_egoview"] = "robot0_oak_egoview"
    """WBC's default ego camera is OAK-D; Realsense is also available."""
    camera_width: int = 640
    camera_height: int = 480
    visualize: bool = True
    """Open the interactive MuJoCo viewer; use --no-visualize for headless runs."""
    video_dir: Path | None = None
    """Record videos and results here when set; recording is disabled by default."""
    smoke_test: bool = False
    """Hold initial joints instead of contacting a model server."""


def load_source_module(name: str, path: Path):
    """Import a dependency's source module without its optional package initializers."""
    if name in sys.modules:
        if Path(sys.modules[name].__file__).resolve() != path.resolve():
            raise RuntimeError(f"{name} was already loaded from a different checkout")
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        del sys.modules[name]
        raise
    return module


def load_g1_model(wbc_path: Path):
    """Import WBC's robot implementation without importing its RoboCasa fork.

    Both checkouts name their package ``robocasa``. Load just the two model
    modules under private names, and give each module its own asset-root alias.
    The active tabletop package and its asset directory remain intact.
    """
    from robocasa.environments.tabletop.tabletop import _ROBOT_POS_OFFSETS
    from robosuite.models.robots.robot_model import REGISTERED_ROBOTS
    from robosuite.robots import register_robot_class
    from robosuite.utils.mjcf_utils import find_elements, new_element

    wbc_path = wbc_path.expanduser().resolve()
    models = wbc_path / "decoupled_wbc/dexmg/gr00trobocasa/robocasa/models"
    if not (models / "robots/manipulators/g1_robot.py").is_file():
        raise FileNotFoundError(
            f"G1 model missing under {models}. Set --wbc-path to the WBC checkout."
        )
    # Import these data-only modules directly: robot_model/__init__.py also
    # imports Pinocchio, which this MuJoCo joint-control experiment does not use.
    supplemental = wbc_path / "decoupled_wbc/control/robot_model/supplemental_info"
    load_source_module(
        "decoupled_wbc.control.robot_model.supplemental_info.robot_supplemental_info",
        supplemental / "robot_supplemental_info.py",
    )
    info_module = load_source_module(
        "decoupled_wbc.control.robot_model.supplemental_info.g1.g1_supplemental_info",
        supplemental / "g1/g1_supplemental_info.py",
    )
    info = info_module.G1SupplementalInfo()
    if "G1Tabletop" in REGISTERED_ROBOTS:
        return info

    modules = []
    for name, relative_path in (
        ("_gr00t_tabletop_g1_hands", "grippers/g1_threefinger_hands.py"),
        ("_gr00t_tabletop_g1_robot", "robots/manipulators/g1_robot.py"),
    ):
        module = load_source_module(name, models / relative_path)
        module.robocasa = SimpleNamespace(
            models=SimpleNamespace(assets_root=str(models / "assets"))
        )
        modules.append(module)

    @register_robot_class("LeggedRobot")
    class G1Tabletop(modules[-1].G1FixedLowerBody):
        @property
        def default_base(self):
            # WBC calls this NullBase; upstream robosuite 1.5.1 names its
            # stationary MobileBaseModel NoActuationBase.
            return "NoActuationBase"

        @property
        def init_qpos(self):
            qpos = super().init_qpos.copy()
            # G1's all-zero pose puts its wrists below the 0.92 m tabletop.
            # Slightly spread the arms and lift the forearms before settling.
            for side, roll in (("left", 0.2), ("right", -0.2)):
                for joint, angle in (("shoulder_roll", roll), ("elbow", -0.5)):
                    name = f"{self.naming_prefix}{side}_{joint}_joint"
                    qpos[self.joints.index(name)] = angle
            return qpos

        def __init__(self, idn=0):
            super().__init__(idn=idn)
            # Reuse WBC's calibrated G1 cameras. Tabletop's default egoview
            # targets GR1's head_pitch body, which G1 does not have.
            for name, camera in self.get_camera_configs().items():
                parent = find_elements(
                    self.root, "body", {"name": camera["parent_body"]}, return_first=True
                )
                parent.append(
                    new_element(
                        "camera",
                        name=name,
                        pos=camera["pos"],
                        quat=camera["quat"],
                        **camera["camera_attribs"],
                    )
                )

    # The tabletop fork's G1 entry has z=0 and adds 0.33 m of clearance,
    # assuming a mobile robot. Our pelvis must instead be at WBC's standing
    # height, close enough to reach the table with the lower body fixed.
    _ROBOT_POS_OFFSETS["G1Tabletop"] = [0, 0, 0.793]
    return info


def make_controller_config():
    """Reuse robosuite's bimanual joint controller with absolute hand targets."""
    from robosuite.controllers import load_composite_controller_config

    config = load_composite_controller_config(robot="GR1")
    config["type"] = "BASIC"
    config["composite_controller_specific_configs"] = {}
    # G1FixedLowerBody has two 7-DoF arms, two hands, and a 3-DoF torso.
    config["body_parts"] = {
        name: part
        for name, part in config["body_parts"].items()
        if name in ("left", "right", "torso")
    }
    for part in config["body_parts"].values():
        part["input_type"] = "absolute"
        if "gripper" in part:
            part["gripper"]["use_action_scaling"] = False
    return config


def make_joint_hold_action(env):
    """Hold current absolute joints in controller/actuator order during reset."""
    robot = env.robots[0]
    sim = env.sim
    parts = {}
    for part, controller in robot.composite_controller.part_controllers.items():
        if part.endswith("gripper"):
            actuators = robot._ref_actuators_indexes_dict[part]
            joint_ids = sim.model.actuator_trnid[actuators, 0]
        else:
            joint_ids = controller.joint_index
        qpos_ids = [
            sim.model.get_joint_qpos_addr(sim.model.joint_id2name(joint)) for joint in joint_ids
        ]
        parts[part] = sim.data.qpos[qpos_ids].copy()
    return robot.create_action_vector(parts)


def wrist_eef_in_base(sim, base_body: str, wrist_body: str) -> np.ndarray:
    """XYZ + row-major first two rotation rows in the G1 pelvis frame."""
    base_rot = sim.data.get_body_xmat(base_body)
    wrist_rot = sim.data.get_body_xmat(wrist_body)
    pos = base_rot.T @ (sim.data.get_body_xpos(wrist_body) - sim.data.get_body_xpos(base_body))
    rot = base_rot.T @ wrist_rot
    return np.concatenate((pos, rot[:2].reshape(6))).astype(np.float32)


class RealG1TabletopEnv(gym.Env):
    """Thin adapter from a RoboCasa G1 scene to GR00T's flat sim interface."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 20}

    def __init__(
        self,
        env,
        robot_info,
        instruction: str | None = None,
        camera_name="robot0_oak_egoview",
        visualize: bool = False,
    ):
        self.env = env
        self.robot_info = robot_info
        self.instruction = instruction
        self.camera_name = camera_name
        self.visualize = visualize
        self.viewer_closed = False
        self.render_mode = "rgb_array"
        self.render_cache = None
        self.observation_space = gym.spaces.Dict(
            {
                **{
                    f"state.{k}": gym.spaces.Box(-np.inf, np.inf, (d,), np.float32)
                    for k, d in STATE_DIMS.items()
                },
                "video.ego_view": gym.spaces.Box(
                    0, 255, (env.camera_heights[0], env.camera_widths[0], 3), np.uint8
                ),
                LANGUAGE_KEY: gym.spaces.Text(
                    1024, charset="".join(chr(i) for i in range(32, 127))
                ),
            }
        )
        self.action_space = gym.spaces.Dict(
            {
                f"action.{k}": gym.spaces.Box(-np.inf, np.inf, (d,), np.float32)
                for k, d in ACTION_DIMS.items()
            }
        )
        self._bind_joints()

    def _bind_joints(self):
        robot = self.env.robots[0]
        model = self.env.sim.model
        prefix = robot.robot_model.naming_prefix
        self.base_body = prefix + self.robot_info.root_frame_name
        self.wrist_bodies = {
            side: prefix + name for side, name in self.robot_info.hand_frame_names.items()
        }
        self.joint_names = {}
        self.qpos_indices = {}
        self.joint_limits = {}
        for group, dim in JOINT_DIMS.items():
            # WBC groups specify policy order. Hand XML/controller order is
            # different (thumb first), so never map hands by array position.
            group_prefix = (
                robot.gripper[group.split("_")[0]].naming_prefix
                if group.endswith("hand")
                else prefix
            )
            names = [group_prefix + name for name in self.robot_info.joint_groups[group]["joints"]]
            if len(names) != dim:
                raise ValueError(f"{group} must have {dim} joints; got {names}")
            self.joint_names[group] = names
            self.qpos_indices[group] = [model.get_joint_qpos_addr(name) for name in names]
            self.joint_limits[group] = np.array(
                [model.jnt_range[model.joint_name2id(name)] for name in names]
            )

        self.part_joint_names = {}
        for part, controller in robot.composite_controller.part_controllers.items():
            group = {
                "left": "left_arm",
                "right": "right_arm",
                "left_gripper": "left_hand",
                "right_gripper": "right_hand",
                "torso": "waist",
            }.get(part)
            if group is None:
                raise ValueError(
                    f"Unexpected controller {part!r}; this experiment requires fixed lower body."
                )
            if part.endswith("gripper"):
                ids = [model.actuator_trnid[i, 0] for i in robot._ref_actuators_indexes_dict[part]]
            else:
                if controller.input_type != "absolute":
                    raise ValueError(f"{part} must consume absolute joint positions.")
                ids = controller.joint_index
            names = [model.joint_id2name(i) for i in ids]
            if set(names) != set(self.joint_names[group]):
                raise ValueError(f"Joint mismatch for {part}: {names} vs {self.joint_names[group]}")
            self.part_joint_names[part] = (
                group,
                [self.joint_names[group].index(name) for name in names],
            )

    def _observation(self, raw_obs):
        sim = self.env.sim
        # robosuite's default image convention is OpenGL (bottom-up).
        self.render_cache = np.ascontiguousarray(raw_obs[self.camera_name + "_image"][::-1])
        obs = {
            f"state.{group}": sim.data.qpos[indexes].astype(np.float32).copy()
            for group, indexes in self.qpos_indices.items()
        }
        for side, body in self.wrist_bodies.items():
            obs[f"state.{side}_wrist_eef_9d"] = wrist_eef_in_base(sim, self.base_body, body)
        obs["video.ego_view"] = self.render_cache
        obs[LANGUAGE_KEY] = self.instruction or self.env.get_ep_meta()["lang"]
        return obs

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            # RoboCasa uses its Generator as well as legacy numpy/random samplers.
            import random

            random.seed(seed)
            np.random.seed(seed)
            self.env.rng = np.random.default_rng(seed)
        raw_obs = self.env.reset()
        self._bind_joints()
        self.viewer_closed = False
        if self.visualize:
            # Frame this robot and its table, rather than the kitchen-wide
            # camera inherited from RoboCasa. The free camera stays interactive.
            pos = self.env.sim.data.get_body_xpos(self.base_body)
            rot = self.env.sim.data.get_body_xmat(self.base_body)
            lookat = pos + rot @ np.array([0.45, 0.0, 0.1])
            self.env.viewer.camera_config = {
                "lookat": lookat,
                "distance": 2.5,
                "azimuth": np.degrees(np.arctan2(rot[1, 0], rot[0, 0])) + 135,
                "elevation": -25,
            }
            # Show the initial scene before waiting for the first model action.
            self.env.viewer.update()
        return self._observation(raw_obs), {"success": False}

    def step(self, action):
        step_start = time.monotonic() if self.visualize else None
        targets = {}
        for group, dim in JOINT_DIMS.items():
            value = np.asarray(action[f"action.{group}"], dtype=np.float64)
            if value.shape != (dim,) or not np.all(np.isfinite(value)):
                raise ValueError(f"Invalid {group} target: expected finite ({dim},), got {value}")
            limits = self.joint_limits[group]
            targets[group] = np.clip(value, limits[:, 0], limits[:, 1])
        parts = {
            part: targets[group][order] for part, (group, order) in self.part_joint_names.items()
        }
        # The model emits absolute joints after decoding. EEF / navigation /
        # base-height outputs are intentionally unused in this tabletop mode.
        vector = self.env.robots[0].create_action_vector(parts)
        raw_obs, reward, done, info = self.env.step(vector)
        info["success"] = bool(self.env._check_success())
        if self.visualize:
            # robosuite's mjviewer syncs on every control step, including the
            # individual steps inside MultiStepWrapper's action chunks.
            self.viewer_closed = not self.env.viewer.viewer.is_running()
            time.sleep(max(0.0, self.env.control_timestep - (time.monotonic() - step_start)))
        return self._observation(raw_obs), float(reward), bool(done), self.viewer_closed, info

    def render(self):
        if self.render_cache is None:
            raise RuntimeError("Call reset() before render().")
        return self.render_cache

    def close(self):
        self.env.close()


def create_env(config: Config) -> RealG1TabletopEnv:
    # Select the pinned tabletop fork explicitly; the kitchen and WBC forks
    # use the same Python package name and are not interchangeable.
    tabletop_path = REPO_ROOT / "external_dependencies/robocasa-gr1-tabletop-tasks"
    sys.path.insert(0, str(tabletop_path))
    import robocasa
    from robocasa.environments.tabletop.tabletop import REGISTERED_TABLETOP_EVNS

    if not Path(robocasa.__file__).resolve().is_relative_to(tabletop_path):
        raise RuntimeError(
            f"Already imported a different RoboCasa: {robocasa.__file__}. Use a fresh process."
        )
    if config.task not in REGISTERED_TABLETOP_EVNS:
        raise ValueError(
            f"Unknown tabletop task {config.task!r}. Available: {sorted(REGISTERED_TABLETOP_EVNS)}"
        )
    robot_info = load_g1_model(config.wbc_path)

    class G1TabletopTask(REGISTERED_TABLETOP_EVNS[config.task]):
        def _reset_internal(self):
            self._holding_initial_pose = True
            try:
                super()._reset_internal()
            finally:
                self._holding_initial_pose = False

        def _pre_action(self, action, policy_step=False):
            # RoboCasa settles objects using a zero delta action. Our joint
            # controllers consume absolute angles, so zero would drive the
            # lifted arms into the table before the first policy observation.
            if self._holding_initial_pose and policy_step:
                action = make_joint_hold_action(self)
            super()._pre_action(action, policy_step)

        def compute_robot_base_placement_pose(self, offset=None):
            offset = np.zeros(3) if offset is None else np.array(offset, dtype=float)
            # Tabletop already leaves 0.20 m between the base and table edge.
            offset[1] += 0.20 - config.robot_table_distance
            pos, ori = super().compute_robot_base_placement_pose(offset)
            pos[2] = config.robot_base_height
            return pos, ori

        def _load_model(self):
            # PnP tasks supply distractor_config explicitly, so passing another
            # value to their constructors raises a duplicate-keyword error.
            # Select the experiment's clutter setting before scene creation.
            self.use_distractors = config.use_distractors
            if not config.use_distractors:
                self.distractor_config = None
            super()._load_model()

        def edit_model_xml(self, xml_str):
            # Tabletop rewrites every /robocasa/ asset to its own checkout.
            # Restore G1 mesh paths after that processing; scene assets still
            # come from the tabletop checkout.
            root = ET.fromstring(super().edit_model_xml(xml_str))
            local_g1 = Path(robocasa.models.assets_root) / "robots/unitree_g1"
            wbc_g1 = config.wbc_path.expanduser().resolve() / (
                "decoupled_wbc/dexmg/gr00trobocasa/robocasa/models/assets/robots/unitree_g1"
            )
            for element in root.findall("./asset/*"):
                file = element.get("file")
                if file is not None and Path(file).is_relative_to(local_g1):
                    element.set("file", str(wbc_g1 / Path(file).relative_to(local_g1)))
            return ET.tostring(root, encoding="unicode")

    env = G1TabletopTask(
        robots="G1Tabletop",
        controller_configs=make_controller_config(),
        camera_names=[config.camera_name],
        camera_widths=config.camera_width,
        camera_heights=config.camera_height,
        use_camera_obs=True,
        has_renderer=config.visualize,
        has_offscreen_renderer=True,
        renderer="mjviewer" if config.visualize else "mujoco",
        render_camera=None,
        control_freq=20,
        horizon=config.max_episode_steps,
        ignore_done=True,
        initialization_noise=None,
        seed=config.seed,
        translucent_robot=False,
        layout_ids=config.layout_id,
        style_ids=config.style_id,
    )
    try:
        return RealG1TabletopEnv(
            env, robot_info, config.instruction, config.camera_name, visualize=config.visualize
        )
    except Exception:
        env.close()
        raise


def validate_modality_config(modalities):
    for modality, expected in (
        ("video", {"ego_view"}),
        ("state", set(STATE_DIMS)),
        ("action", set(ACTION_DIMS)),
        ("language", {LANGUAGE_KEY}),
    ):
        actual = set(modalities[modality].modality_keys)
        if actual != expected:
            raise ValueError(
                f"Server is not configured for REAL_G1: {modality} keys {actual}; expected {expected}."
            )


def main(config: Config):
    for name in (
        "n_episodes",
        "max_episode_steps",
        "n_action_steps",
        "camera_width",
        "camera_height",
        "robot_base_height",
        "robot_table_distance",
    ):
        if getattr(config, name) <= 0:
            raise ValueError(f"{name} must be positive")
    os.environ.setdefault("MUJOCO_GL", "egl")
    with ExitStack() as stack:
        policy = None
        if config.smoke_test:
            # No model loading for the simulator-only check. Real evaluation
            # always resolves temporal offsets from the server's checkpoint.
            modalities = {
                "video": ModalityConfig(delta_indices=[-20, 0], modality_keys=["ego_view"]),
                "state": ModalityConfig(delta_indices=[0], modality_keys=list(STATE_DIMS)),
                "action": ModalityConfig(
                    delta_indices=list(range(40)), modality_keys=list(ACTION_DIMS)
                ),
                "language": ModalityConfig(delta_indices=[0], modality_keys=[LANGUAGE_KEY]),
            }
        else:
            from gr00t.policy.server_client import PolicyClient

            policy = stack.enter_context(
                PolicyClient(
                    host=config.policy_host,
                    port=config.policy_port,
                    timeout_ms=config.policy_timeout_ms,
                )
            )
            modalities = policy.get_modality_config()
            validate_modality_config(modalities)
        contract = PolicyHorizonSpec.from_modality_config(
            modalities, n_action_steps=config.n_action_steps
        )
        env = create_env(config)
        stack.callback(lambda: env.close())
        if config.video_dir is not None:
            env = VideoRecordingWrapper(
                env,
                video_dir=config.video_dir,
                steps_per_render=1,
                max_episode_steps=config.max_episode_steps,
                fps=20,
                record_video_keys=("video.ego_view",),
            )
        env = MultiStepWrapper(
            env,
            contract=contract,
            max_episode_steps=config.max_episode_steps,
            terminate_on_success=True,
        )
        results = []
        for episode in range(config.n_episodes):
            obs, _ = env.reset(seed=config.seed + episode)
            if policy is not None:
                policy.reset()
            held_joints = {group: obs[f"state.{group}"][-1].copy() for group in JOINT_DIMS}
            success, steps, total_reward = False, 0, 0.0
            while True:
                if policy is None:
                    actions = {
                        f"action.{group}": np.repeat(value[None], contract.action_horizon, axis=0)
                        for group, value in held_joints.items()
                    }
                else:
                    batched_obs = {
                        key: [value] if isinstance(value, str) else value[None]
                        for key, value in obs.items()
                    }
                    batched_actions, _ = policy.get_action(batched_obs)
                    actions = {}
                    for key, dim in ACTION_DIMS.items():
                        value = np.asarray(batched_actions[f"action.{key}"])
                        if value.shape != (1, contract.action_horizon, dim) or not np.all(
                            np.isfinite(value)
                        ):
                            raise ValueError(f"Invalid action.{key} chunk: {value.shape}")
                        actions[f"action.{key}"] = value[0]
                obs, _, terminated, truncated, info = env.step(actions)
                steps += int(info["n_env_steps"])
                # MultiStepWrapper's returned reward aggregates the entire
                # episode so far; sum only this chunk's per-step rewards.
                total_reward += float(np.sum(info["rewards"]))
                success |= bool(np.any(info["success"]))
                if terminated or truncated:
                    break
            result = {
                "episode": episode,
                "seed": config.seed + episode,
                "success": success,
                "steps": steps,
                "reward": total_reward,
            }
            results.append(result)
            print(json.dumps(result))
            if env.unwrapped.viewer_closed:
                break
        summary = {
            "task": config.task,
            "robot": "G1Tabletop",
            "embodiment": "REAL_G1",
            "smoke_test": config.smoke_test,
            "use_distractors": config.use_distractors,
            "robot_base_height": config.robot_base_height,
            "robot_table_distance": config.robot_table_distance,
            "camera_name": config.camera_name,
            "episodes": results,
        }
        if not config.smoke_test:
            summary["success_rate"] = sum(r["success"] for r in results) / len(results)
        if config.video_dir is not None:
            (config.video_dir / "results.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main(tyro.cli(Config))
