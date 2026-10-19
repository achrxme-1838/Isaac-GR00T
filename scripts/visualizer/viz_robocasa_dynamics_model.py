"""Replay recorded robot states with a green, translucent predicted EEF.

Edit CONFIG below, then run from Isaac-GR00T:
    MUJOCO_GL=egl gr00t/eval/sim/robocasa/robocasa_uv/.venv/bin/python \
        scripts/visualizer/viz_robocasa_dynamics_model.py

No policy server or W&B run is needed. Actions enter only the dynamics model.
The simulator receives qpos directly and never integrates physics during playback.
"""

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from uuid import uuid4


REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
# Set this before importing RoboSuite/OpenGL, including when recording without X11.
os.environ.setdefault("MUJOCO_GL", "egl")

from gr00t.eval.sim.gui_video_recorder import GuiVideoRecorder  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402


@dataclass
class VizConfig:
    dynamics_repo: Path = REPO.parent / "mobile_crossemb"
    # Relative checkpoint paths are resolved against dynamics_repo, independent of cwd.
    checkpoint_path: str = "outputs/source_dynamics/20261019T103136Z_b44c4794/best.pt"
    data_path: str | None = None  # None: use the checkpoint's dataset.
    split: str = "val"
    num_trajectories: int = 10  # One sampled start per distinct trajectory.
    rollout_steps: int = 200 #50  # Recursion steps; history comes from the checkpoint.
    seed: int = 42
    device: str = "auto"
    num_threads: int = 4
    repeats: int = 1  # Replay the sampled collection this many times.
    playback_speed: float = 1.0
    pause_seconds: float = 1.0  # Hold the initial and final frame in the GUI.
    env_name: str = "robocasa_panda_omron/OpenDrawer_PandaOmron_Env"
    scene_seed: int = 0  # Scene is illustrative: dataset contains no scene qpos/XML.
    ghost_rgba: tuple = (0.1, 1.0, 0.1, 0.35)
    record_video: bool = True  # True: save MP4s without opening a GUI.
    record_camera: str = "robot0_agentview_left"
    output_dir: Path = REPO / "outputs/dynamics_visualizer"

    def validate(self):
        for name in ("num_trajectories", "rollout_steps", "repeats", "num_threads"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer.")
        if not math.isfinite(self.playback_speed) or self.playback_speed <= 0:
            raise ValueError("playback_speed must be finite and positive.")
        if not math.isfinite(self.pause_seconds) or self.pause_seconds < 0:
            raise ValueError("pause_seconds must be finite and nonnegative.")
        rgba = np.asarray(self.ghost_rgba)
        if rgba.shape != (4,) or not np.isfinite(rgba).all() or np.any((rgba < 0) | (rgba > 1)):
            raise ValueError("ghost_rgba must contain four values in [0, 1].")
        if not 0 < rgba[3] < 1:
            raise ValueError("Use a translucent ghost alpha in (0, 1).")
        if not self.env_name.startswith("robocasa_panda_omron/"):
            raise ValueError("This replay currently supports PandaOmron environments.")


CONFIG = VizConfig()


def field(state, mapping, name):
    span = mapping[name]
    return np.asarray(state[span.start : span.end], dtype=np.float64)


def prepare_rollouts(cfg, output):
    """Use the evaluator's sampler, history, action-prefix rules, and feedback unchanged."""
    root = Path(cfg.dynamics_repo).expanduser().resolve()
    if not (root / "source/evaluation/rollout.py").is_file():
        raise FileNotFoundError(f"Set dynamics_repo to mobile_crossemb: {root}")
    sys.path.insert(0, str(root))
    from source.config import DatasetConfig
    from source.data.schema import load_state_mapping, select_indices, vector_dim
    from source.evaluation.config import EvalConfig
    from source.evaluation.feedback import StateFeedback
    from source.evaluation.rollout import recursion_stride, rollout_trajectory, sample_trajectories
    from source.training import load_dynamics
    import torch

    checkpoint = Path(cfg.checkpoint_path).expanduser()
    checkpoint = (checkpoint if checkpoint.is_absolute() else root / checkpoint).resolve()
    evaluation = EvalConfig(
        checkpoint_path=checkpoint,
        split=cfg.split,
        num_trajectories=cfg.num_trajectories,
        rollout_steps=cfg.rollout_steps,
        seed=cfg.seed,
    )
    evaluation.validate()
    torch.set_num_threads(cfg.num_threads)
    device = (
        ("cuda" if torch.cuda.is_available() else "cpu") if cfg.device == "auto" else cfg.device
    )
    model, metadata = load_dynamics(checkpoint, device=device)
    data_cfg = DatasetConfig(**metadata["config"]["data"])
    data_root = Path(cfg.data_path or data_cfg.path).expanduser()
    data_cfg.path = (data_root if data_root.is_absolute() else root / data_root).resolve()
    info = json.loads((data_cfg.path / "meta/info.json").read_text())
    if "panda" not in info.get("robot_type", "").lower():
        raise ValueError("The replay geometry requires a Panda dataset.")
    mapping = load_state_mapping(data_cfg, info)
    for name, width in {
        "base_pos_w": 3,
        "base_quat_w": 4,
        "joint_pos": 7,
        "gripper_pos": 2,
        "ee_pos_w": 3,
        "ee_quat_w": 4,
    }.items():
        if name not in mapping or mapping[name].dim != width:
            raise ValueError(f"Replay requires {name} with dimension {width}.")
    inputs = select_indices(data_cfg.sub_state_space, mapping)
    targets = select_indices(data_cfg.next_sub_state_space, mapping)
    resolved = metadata["resolved"]
    if inputs != resolved["input_indices"] or targets != resolved["target_indices"]:
        raise ValueError("Dataset state mapping differs from the checkpoint.")
    if vector_dim(info, data_cfg.action_column) != model.action_dim:
        raise ValueError("Dataset action dimension differs from the checkpoint.")
    stride = recursion_stride(data_cfg)
    fps = float(info["fps"])
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("Dataset fps must be finite and positive.")
    sampled = sample_trajectories(data_cfg, info, resolved, evaluation, stride)
    clips = []
    with (output / "feedback_warnings.jsonl").open("w") as warnings_file:

        def report(event, stage="feedback"):
            event = {"stage": stage, **event}
            print(f"WARNING {json.dumps(event)}", flush=True)
            warnings_file.write(json.dumps(event) + "\n")
            warnings_file.flush()

        for index, (trajectory, start) in enumerate(sampled):
            initial = trajectory.states[start]
            feedback = StateFeedback(
                data_cfg.sub_state_space,
                data_cfg.next_sub_state_space,
                mapping,
                initial,
                info["robot_type"],
                evaluation.kinematics,
            )
            result = rollout_trajectory(
                model,
                data_cfg,
                mapping,
                trajectory,
                start,
                cfg.rollout_steps,
                inputs,
                targets,
                feedback,
                device,
                report,
            )
            frames = start + np.arange(cfg.rollout_steps + 1) * stride
            recorded = trajectory.states[frames].copy()
            # Display completion is separate from recurrent feedback. In particular,
            # converting a local prediction to world cannot inject data into the model.
            display = StateFeedback(
                ("ee_pos_w", "ee_quat_w"),
                data_cfg.next_sub_state_space,
                mapping,
                initial,
                info["robot_type"],
                evaluation.kinematics,
            )
            poses = [
                np.r_[field(initial, mapping, "ee_pos_w"), field(initial, mapping, "ee_quat_w")]
            ]
            previous = initial
            for step, prediction in enumerate(result["predictions"], 1):
                completed, event = display.complete(prediction, recorded[step], previous)
                pose = np.r_[
                    field(completed, mapping, "ee_pos_w"), field(completed, mapping, "ee_quat_w")
                ]
                if not np.isfinite(pose).all() or np.linalg.norm(pose[3:]) < 1e-8:
                    raise ValueError(f"Invalid predicted EEF pose at step {step}.")
                pose[3:] /= np.linalg.norm(
                    pose[3:]
                )  # Render rotation only; leave model outputs intact.
                poses.append(pose)
                if event["dataset_fields"] or event["reconstructed"]:
                    report(
                        {
                            **event,
                            "episode_index": trajectory.episode_index,
                            "recursion_step": step,
                            "frame": int(frames[step]),
                        },
                        "display",
                    )
                previous = completed
            clip = dict(
                episode_index=trajectory.episode_index,
                start_frame=start,
                frames=frames,
                recorded_states=recorded,
                predicted_poses=np.asarray(poses),
            )
            clips.append(clip)
            np.savez_compressed(output / f"trajectory_{index + 1:03d}.npz", **clip, **result)
            print(
                f"Prepared {index + 1}/{len(sampled)}: episode={trajectory.episode_index}, "
                f"start={start}, steps={cfg.rollout_steps}, stride={stride}",
                flush=True,
            )
    manifest = {
        "config": asdict(cfg),
        "checkpoint_path": str(checkpoint),
        "data_path": str(data_cfg.path),
        "training_config": metadata["config"],
        "fps": fps,
        "frame_stride": stride,
        "samples": [
            {"episode_index": c["episode_index"], "start_frame": c["start_frame"]} for c in clips
        ],
        "scene": "Illustrative reset only; dataset has no scene XML/object qpos.",
        "eef": "Grip-site position + right_hand body rotation (xyzw); recorded finger opening.",
    }
    (output / "config.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    return clips, mapping, fps, stride


class RobotStateReplay:
    """Set arm/finger joints and solve the four base DOFs from recorded base pose."""

    def __init__(self, robot_env, mapping):
        self.model, self.data = robot_env.sim.model._model, robot_env.sim.data._data
        self.mapping = mapping
        self.arm = [self.model.joint(f"robot0_joint{i}").qposadr[0] for i in range(1, 8)]
        self.fingers = [
            self.model.joint(f"gripper0_right_finger_joint{i}").qposadr[0] for i in (1, 2)
        ]
        self.slides = [
            self.model.joint(f"mobilebase0_joint_{name}").qposadr[0]
            for name in ("mobile_forward", "mobile_side", "torso_height")
        ]
        self.yaw_joint = self.model.joint("mobilebase0_joint_mobile_yaw").id
        self.yaw = self.model.jnt_qposadr[self.yaw_joint]
        self.base = self.model.site("mobilebase0_center").id
        self.grip = self.model.site("gripper0_right_grip_site").id
        self.hand = self.model.body("robot0_right_hand").id

    def set_state(self, state, timestamp):
        m, d = self.model, self.data
        if not np.isfinite(state).all():
            raise ValueError("Cannot replay a non-finite recorded state.")
        d.qpos[self.arm] = field(state, self.mapping, "joint_pos")
        d.qpos[self.fingers] = field(state, self.mapping, "gripper_pos")
        d.qpos[self.slides] = 0
        d.qpos[self.yaw] = 0
        mujoco.mj_kinematics(m, d)
        desired = Rotation.from_quat(field(state, self.mapping, "base_quat_w"))
        zero_rotation = Rotation.from_matrix(d.site_xmat[self.base].reshape(3, 3))
        rotation_delta = (desired * zero_rotation.inv()).as_rotvec()
        axis = d.xaxis[self.yaw_joint].copy()
        if np.linalg.norm(rotation_delta - axis * np.dot(rotation_delta, axis)) > 1e-4:
            raise ValueError("Recorded base has roll/pitch unsupported by the Omron yaw joint.")
        d.qpos[self.yaw] = np.dot(rotation_delta, axis)
        mujoco.mj_kinematics(m, d)
        origin = d.site_xpos[self.base].copy()
        # A slide changes position linearly. Measuring the axes also handles a
        # randomly rotated RoboCasa scene root and the yaw joint's offset pivot.
        axes = []
        for address in self.slides:
            d.qpos[address] = 1
            mujoco.mj_kinematics(m, d)
            axes.append(d.site_xpos[self.base].copy() - origin)
            d.qpos[address] = 0
        d.qpos[self.slides] = np.linalg.solve(
            np.column_stack(axes), field(state, self.mapping, "base_pos_w") - origin
        )
        d.qvel[:] = 0
        d.ctrl[:] = 0
        d.qacc_warmstart[:] = 0
        d.time = timestamp
        mujoco.mj_forward(m, d)  # Refresh transforms only: NEVER mj_step/env.step.
        position_error = np.linalg.norm(
            d.site_xpos[self.grip] - field(state, self.mapping, "ee_pos_w")
        )
        hand_rotation = Rotation.from_matrix(d.xmat[self.hand].reshape(3, 3))
        rotation_error = (
            hand_rotation.inv() * Rotation.from_quat(field(state, self.mapping, "ee_quat_w"))
        ).magnitude()
        if position_error > 1e-3 or rotation_error > 1e-3:
            raise ValueError(
                f"Replay/model convention mismatch: {position_error:.6g} m, {rotation_error:.6g} rad."
            )
        return position_error, rotation_error


class EefGhost:
    """Viewer-only hand/gripper meshes moved to a predicted grip position/body rotation.

    Like NominalArmGhost, decoration geoms reuse the live model's mesh assets.
    No arm IK is needed to display an EEF, including unreachable predictions.
    """

    def __init__(self, robot_env, rgba):
        self.model, self.data = robot_env.sim.model._model, robot_env.sim.data._data
        self.hand = self.model.body("robot0_right_hand").id
        self.grip = self.model.site("gripper0_right_grip_site").id
        descendants = {self.hand}
        for body in range(self.hand + 1, self.model.nbody):
            if self.model.body_parentid[body] in descendants:
                descendants.add(body)
        self.geom_ids = [
            i
            for i in range(self.model.ngeom)
            if self.model.geom_group[i] == 1 and self.model.geom_bodyid[i] in descendants
        ]
        if not self.geom_ids:
            raise ValueError("No Panda EEF visual geometry found.")
        self.rgba = np.asarray(rgba, dtype=np.float32)

    def update(self, scene, pose, *, start=0):
        pose = np.asarray(pose, dtype=np.float64)
        if pose.shape != (7,) or not np.isfinite(pose).all() or np.linalg.norm(pose[3:]) < 1e-8:
            raise ValueError("EEF pose must contain position + nonzero xyzw quaternion.")
        if start + len(self.geom_ids) > scene.maxgeom:
            raise ValueError("Viewer scene has insufficient EEF geometry capacity.")
        delta = Rotation.from_quat(pose[3:]).as_matrix() @ self.data.xmat[self.hand].reshape(3, 3).T
        anchor = self.data.site_xpos[self.grip]
        scene.ngeom = start
        for geom_id in self.geom_ids:
            geom = scene.geoms[scene.ngeom]
            mujoco.mjv_initGeom(
                geom,
                type=int(self.model.geom_type[geom_id]),
                size=self.model.geom_size[geom_id],
                pos=pose[:3] + delta @ (self.data.geom_xpos[geom_id] - anchor),
                mat=(delta @ self.data.geom_xmat[geom_id].reshape(3, 3)).ravel(),
                rgba=self.rgba,
            )
            geom.dataid = (
                2 * self.model.geom_dataid[geom_id]
                if geom.type == mujoco.mjtGeom.mjGEOM_MESH
                else -1
            )
            geom.category = mujoco.mjtCatBit.mjCAT_DECOR
            geom.objtype = mujoco.mjtObj.mjOBJ_UNKNOWN
            geom.objid = geom.segid = geom.matid = -1
            geom.emission, geom.specular = 0.2, 0.2
            geom.modelrbound = self.model.geom_rbound[geom_id]
            geom.transparent = True
            scene.ngeom += 1


class EefVideoRecorder(GuiVideoRecorder):
    def write_pose(self, ghost, pose):
        self.renderer.update_scene(self.data, camera=self.camera, scene_option=self.option)
        ghost.update(self.renderer.scene, pose, start=self.renderer.scene.ngeom)
        try:
            self.process.stdin.write(np.ascontiguousarray(self.renderer.render()).tobytes())
        except BrokenPipeError:
            self.close()
            raise


def hold(viewer, seconds):
    deadline = time.perf_counter() + seconds
    while viewer.is_running() and time.perf_counter() < deadline:
        viewer.sync()
        time.sleep(min(0.02, max(0, deadline - time.perf_counter())))
    return viewer.is_running()


def visualize(cfg=CONFIG):
    cfg.validate()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = Path(cfg.output_dir).expanduser().resolve() / f"{stamp}_{uuid4().hex[:8]}"
    output.mkdir(parents=True)
    print(f"Output: {output}", flush=True)
    clips, mapping, fps, stride = prepare_rollouts(cfg, output)
    print(
        "Robot: recorded states. Green EEF: recursive prediction. "
        "Scene is illustrative (no recorded scene XML/object states).",
        flush=True,
    )

    import gymnasium as gym
    import robocasa.utils.gym_utils.gymnasium_groot  # noqa: F401

    env = gym.make(
        cfg.env_name,
        enable_render=cfg.record_video,
        seed=cfg.scene_seed,
        camera_widths=128,
        camera_heights=128,
    )
    viewer = recorder = None
    errors = []
    try:
        # One static scene for the collection. Playback never calls env.step().
        env.reset(seed=cfg.scene_seed)
        robot_env = env.unwrapped.env
        replay = RobotStateReplay(robot_env, mapping)
        ghost = EefGhost(robot_env, cfg.ghost_rgba)
        if not cfg.record_video:
            import mujoco.viewer as mj_viewer

            viewer = mj_viewer.launch_passive(replay.model, replay.data)
            with viewer.lock():
                viewer.opt.geomgroup[0] = 0
                viewer.opt.geomgroup[1] = 1
                viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
                viewer.cam.fixedcamid = replay.model.camera(cfg.record_camera).id
        period = stride / (fps * cfg.playback_speed)
        for repeat in range(cfg.repeats):
            for index, clip in enumerate(clips):
                print(
                    f"Replay {repeat + 1}/{cfg.repeats}, trajectory {index + 1}/{len(clips)}: "
                    f"episode={clip['episode_index']}, start={clip['start_frame']}",
                    flush=True,
                )
                if cfg.record_video:
                    # Recorder only needs these attributes; fps follows the displayed time stride.
                    recording_env = SimpleNamespace(
                        sim=robot_env.sim,
                        control_freq=1 / period,
                        render_collision_mesh=False,
                        render_visual_mesh=True,
                    )
                    recorder = EefVideoRecorder(
                        recording_env,
                        output / f"repeat_{repeat + 1:02d}_trajectory_{index + 1:03d}.mp4",
                        camera=cfg.record_camera,
                    )
                for step, (frame, state, pose) in enumerate(
                    zip(clip["frames"], clip["recorded_states"], clip["predicted_poses"])
                ):
                    began = time.perf_counter()
                    if viewer is not None:
                        if not viewer.is_running():
                            return output
                        with viewer.lock():
                            errors.append(replay.set_state(state, frame / fps))
                            ghost.update(viewer.user_scn, pose)
                        viewer.sync()
                        delay = (
                            cfg.pause_seconds
                            if step == 0
                            else max(0, period - (time.perf_counter() - began))
                        )
                        if not hold(viewer, delay):
                            return output
                    else:
                        errors.append(replay.set_state(state, frame / fps))
                        recorder.write_pose(ghost, pose)
                if recorder is not None:
                    recorder.close()
                    recorder = None
                if viewer is not None and not hold(viewer, cfg.pause_seconds):
                    return output
    except KeyboardInterrupt:
        print("Visualization stopped.", flush=True)
    finally:
        try:
            if recorder is not None:
                recorder.close()
        finally:
            try:
                if viewer is not None:
                    viewer.close()
            finally:
                env.close()
        if errors:
            stats = {
                "frames_replayed": len(errors),
                "max_replay_position_error_m": float(np.max(errors, axis=0)[0]),
                "max_replay_rotation_error_rad": float(np.max(errors, axis=0)[1]),
            }
            (output / "replay_validation.json").write_text(json.dumps(stats, indent=2) + "\n")
    print(f"Visualization complete: {output}", flush=True)
    return output


if __name__ == "__main__":
    visualize()
