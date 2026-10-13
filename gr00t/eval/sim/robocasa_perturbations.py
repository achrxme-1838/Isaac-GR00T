"""Opt-in physical Panda arm perturbations for RoboCasa GUI evaluation."""

from gr00t.eval.sim.robocasa_exp_adaptation import (
    ARM_JOINTS,
    POSITION_TOLERANCE,
    ROTATION_TOLERANCE,
    PandaKinematics,
)
import gymnasium as gym
import mujoco
import numpy as np


class InitialEESphere:
    """Draw the fixed-distance target sphere in the viewer/recorder's separate scene."""

    def __init__(self, center, target, radius):
        self.center = center.copy()
        self.target = target.copy()
        self.radius = radius

    def update(self, scene, q, *, start=0):
        for index, (pos, radius, rgba) in enumerate(
            (
                (self.center, self.radius, [0.2, 0.7, 1.0, 0.15]),
                (self.target, min(0.005, self.radius / 10), [1.0, 0.5, 0.1, 0.9]),
            ),
            start=start,
        ):
            mujoco.mjv_initGeom(
                scene.geoms[index],
                mujoco.mjtGeom.mjGEOM_SPHERE,
                np.full(3, radius),
                pos,
                np.eye(3).ravel(),
                np.asarray(rgba, dtype=np.float32),
            )
        scene.ngeom = start + 2


class PandaPerturbationWrapper(gym.Wrapper):
    """Change physical state before GUI rendering and temporal observation stacking.

    Only the seven arm joints are affected. Velocity suppression runs at each
    environment control step boundary, including every step in an action chunk,
    rather than at MuJoCo's smaller integration substeps.
    """

    def __init__(self, env, *, initial_ee_pos=0.0, random_joint=0.0, no_vel=False):
        super().__init__(env)
        for value in (initial_ee_pos, random_joint):
            if not np.isfinite(value) or value < 0:
                raise ValueError("Perturbation magnitudes must be finite and nonnegative.")
        self.initial_ee_pos = initial_ee_pos
        self.random_joint = random_joint
        self.no_vel = no_vel
        self.rng = np.random.default_rng()
        self.center = None
        self.target = None

    def reset(self, *, seed=None, **kwargs):
        obs, info = self.env.reset(seed=seed, **kwargs)
        if not (self.initial_ee_pos or self.random_joint or self.no_vel):
            return obs, info
        if seed is not None:
            # Keep task/layout RNG independent of perturbation sampling.
            self.rng = np.random.default_rng(seed)
        robot_env = self.unwrapped.env
        robot = robot_env.robots[0]
        sim = robot_env.sim
        model = sim.model._model
        joint_ids = [model.joint(name).id for name in ARM_JOINTS]
        self.qpos_indices = model.jnt_qposadr[joint_ids]
        self.dof_indices = model.jnt_dofadr[joint_ids]
        lower, upper = model.jnt_range[joint_ids].T
        q = sim.data.qpos[self.qpos_indices].copy()
        site_id = robot.eef_site_id["right"]
        self.center = sim.data.site_xpos[site_id].copy()
        self.target = None
        info = dict(info)
        if self.initial_ee_pos:
            arm = PandaKinematics(robot.robot_model.get_xml(), model.site(site_id).name)
            position, rotation = arm.pose(q)
            root_rotation = sim.data.get_body_xmat("robot0_link0").reshape(3, 3)
            for _ in range(20):
                # Uniform direction on the sphere; always use the full radius.
                direction = self.rng.normal(size=3)
                length = np.linalg.norm(direction)
                if length == 0:
                    continue
                offset = (direction / length) * self.initial_ee_pos
                candidate, error = arm.inverse(position + root_rotation.T @ offset, rotation, q)
                if (
                    np.linalg.norm(error[:3]) <= POSITION_TOLERANCE
                    and np.linalg.norm(error[3:]) <= ROTATION_TOLERANCE
                ):
                    q = candidate
                    self.target = self.center + offset
                    break
            else:
                raise RuntimeError(
                    "Could not find a reachable initial EE position after 20 samples. "
                    "Reduce --initial_ee_pos (radius in metres)."
                )
            info["perturbation_ee_offset_m"] = offset.copy()
            print(f"  Initial EE offset (world, m): {offset.round(5)}", flush=True)
        if self.random_joint:
            offset = self.rng.uniform(-self.random_joint, self.random_joint, len(q))
            perturbed_q = np.clip(q + offset, lower, upper)
            info["perturbation_joint_offset_rad"] = perturbed_q - q
            q = perturbed_q
            print(
                f"  Initial joint offset (rad): {info['perturbation_joint_offset_rad'].round(5)}",
                flush=True,
            )
        if self.initial_ee_pos or self.random_joint:
            sim.data.qpos[self.qpos_indices] = q
        # A relocated initial arm starts at rest; preserve all other DOFs.
        self._zero_velocity()
        if self.initial_ee_pos or self.random_joint:
            controller = robot.part_controllers["right"]
            # OSC also uses the initial joints as its nullspace posture target.
            controller.update_initial_joints(q)
            controller.reset_goal()
        return self._fresh_observation(), info

    def _zero_velocity(self):
        sim = self.unwrapped.env.sim
        sim.data.qvel[self.dof_indices] = 0
        sim.forward()

    def _fresh_observation(self):
        """Rebuild joint/trigonometric/EE observations and images from the new state."""
        groot_env = self.unwrapped
        robot_env = groot_env.env
        robot_env._obs_cache.clear()
        source = robot_env.viewer if robot_env.viewer_get_obs else robot_env
        raw_obs = source._get_observations(force_update=True)
        return groot_env.get_groot_observation(groot_env.get_basic_observation(raw_obs))

    def step(self, action):
        if self.no_vel:
            self._zero_velocity()
        obs, reward, terminated, truncated, info = self.env.step(action)
        if self.no_vel:
            self._zero_velocity()
            obs = self._fresh_observation()
        return obs, reward, terminated, truncated, info

    def make_overlay(self, robot_env):
        return InitialEESphere(self.center, self.target, self.initial_ee_pos)
