"""Sync initial poses using real MuJoCo/controllers without RoboCasa assets or GL."""

from copy import deepcopy

import gymnasium as gym
import numpy as np
import pytest
from scipy.spatial.transform import Rotation


pytest.importorskip("mujoco")
pytest.importorskip("robosuite")

from gr00t.eval.sim.robocasa_exp_adaptation import (  # noqa: E402
    ARM_JOINTS,
    ExpertAdaptationWrapper,
    PandaKinematics,
)
from gr00t.eval.sim.robocasa_sync_init import SyncInitialPoseWrapper  # noqa: E402
from robosuite.controllers import load_composite_controller_config  # noqa: E402
from robosuite.environments.manipulation.lift import Lift  # noqa: E402


class SettlingLift(Lift):
    """Exercise RoboCasa's settle-during-reset pattern using built-in RoboSuite assets."""

    def _load_model(self):
        super()._load_model()
        self.robots[0].init_qpos = [-0.016, -1.034, -0.024, -2.276, 0.039, 1.516, 0.696]
        self.robots[0].robot_model.set_base_ori([0, 0, 0.8])

    def _reset_internal(self):
        super()._reset_internal()
        # Include RNG draws during the deterministic rebuild, as RoboCasa tasks do.
        self.rng.uniform()
        action = np.zeros(self.action_dim)
        for i in range(10):
            self.sim.step1()
            self._pre_action(action, policy_step=i == 0)
            self.sim.step2()


class GrootAdapter(gym.Env):
    """Just the RoboCasa observation API needed by the shared synchronization wrapper."""

    def __init__(self):
        self.env = SettlingLift(
            robots="PandaOmron",
            controller_configs=load_composite_controller_config(
                controller=None, robot="PandaOmron"
            ),
            has_renderer=False,
            has_offscreen_renderer=False,
            use_camera_obs=False,
            seed=5,
        )

    def get_basic_observation(self, obs):
        return obs

    def get_groot_observation(self, obs):
        return {
            "state.joint_position": np.asarray(obs["robot0_joint_pos"], dtype=np.float32),
            "state.joint_position_cos": np.asarray(obs["robot0_joint_pos_cos"], dtype=np.float32),
            "state.joint_position_sin": np.asarray(obs["robot0_joint_pos_sin"], dtype=np.float32),
            "state.joint_velocity": np.asarray(obs["robot0_joint_vel"], dtype=np.float32),
            "state.end_effector_position": np.asarray(obs["robot0_eef_pos"], dtype=np.float32),
        }

    def reset(self, **kwargs):
        raw_obs = self.env.reset()
        site = self.env.robots[0].eef_site_id["right"]
        sim = self.env.sim
        self.nominal_state = sim.get_state()
        self.nominal_pose = (
            sim.data.site_xpos[site].copy(),
            sim.data.site_xmat[site].reshape(3, 3).copy(),
        )
        self.nominal_rng = deepcopy(self.env.rng.bit_generator.state)
        self.nominal_xml = self.env.model.get_xml()
        self.nominal_obs = self.get_groot_observation(raw_obs)
        self.reset_info = {"success": False}
        return self.nominal_obs, self.reset_info

    def step(self, action):
        self.last_action = action
        obs, reward, done, info = self.env.step(action)
        return self.get_groot_observation(obs), reward, done, False, info

    def close(self):
        self.env.close()


@pytest.fixture
def raw():
    env = GrootAdapter()
    try:
        yield env
    finally:
        env.close()


@pytest.mark.parametrize(
    "scales",
    [{"link2": 1.2, "link4": 1.2}, {"link0": 1.2}, {"link7": 1.2}, {"link2": 0.8, "link4": 0.8}],
)
def test_sync_matches_settled_nominal_pose_preserves_scene_and_rebuilds_each_reset(raw, scales):
    wrapped = SyncInitialPoseWrapper(raw, scales)
    for _ in range(2):
        obs, info = wrapped.reset()
        sim = raw.env.sim
        model = sim.model._model
        site = raw.env.robots[0].eef_site_id["right"]
        position, rotation = raw.nominal_pose
        np.testing.assert_allclose(sim.data.site_xpos[site], position, atol=1e-4)
        assert (
            Rotation.from_matrix(sim.data.site_xmat[site].reshape(3, 3) @ rotation.T).magnitude()
            < 1e-3
        )
        assert info["sync_init_position_error_m"] < 1e-4
        assert info["sync_init_rotation_error_rad"] < 1e-3
        ids = [model.joint(name).id for name in ARM_JOINTS]
        qpos_indices, dof_indices = model.jnt_qposadr[ids], model.jnt_dofadr[ids]
        other_pos = np.setdiff1d(np.arange(model.nq), qpos_indices)
        other_vel = np.setdiff1d(np.arange(model.nv), dof_indices)
        np.testing.assert_array_equal(sim.data.qpos[other_pos], raw.nominal_state.qpos[other_pos])
        np.testing.assert_array_equal(sim.data.qvel[other_vel], raw.nominal_state.qvel[other_vel])
        np.testing.assert_array_equal(sim.data.qvel[dof_indices], 0)
        assert raw.env.rng.bit_generator.state == raw.nominal_rng
        assert sim.data.time == raw.nominal_state.time
        assert raw.env.timestep == 0 and raw.env.cur_time == 0
        np.testing.assert_allclose(obs["state.joint_position"], sim.data.qpos[qpos_indices])
        np.testing.assert_allclose(
            obs["state.joint_position_sin"], np.sin(obs["state.joint_position"]), atol=1e-7
        )
        np.testing.assert_allclose(
            obs["state.joint_position_cos"], np.cos(obs["state.joint_position"]), atol=1e-7
        )
        controller = raw.env.robots[0].part_controllers["right"]
        np.testing.assert_allclose(controller.initial_joint, sim.data.qpos[qpos_indices])
        np.testing.assert_allclose(controller.goal_pos, controller.ref_pos)
        np.testing.assert_allclose(controller.goal_ori, controller.ref_ori_mat)
        # Scaling is applied once per rebuild; the scene's source XML stays nominal.
        assert raw.env.model.get_xml() == raw.nominal_xml
        for link, ratio in scales.items():
            child, axis, length = {
                "link0": ("link1", 2, 0.333),
                "link2": ("link3", 1, -0.316),
                "link4": ("link5", 1, 0.384),
                "link7": ("right_hand", 2, 0.1065),
            }[link]
            assert abs(model.body(f"robot0_{child}").pos[axis]) == pytest.approx(
                abs(length * ratio)
            )
        action = np.zeros(raw.env.action_dim)
        wrapped.step(action)
        assert raw.last_action is action


@pytest.mark.parametrize("scales", [{}, {"link2": 1.0, "link4": 1.0}])
def test_unit_ratios_are_exact_identity(raw, scales):
    obs, info = SyncInitialPoseWrapper(raw, scales).reset()
    assert obs is raw.nominal_obs and info is raw.reset_info
    np.testing.assert_array_equal(raw.env.sim.data.qpos, raw.nominal_state.qpos)
    np.testing.assert_array_equal(raw.env.sim.data.qvel, raw.nominal_state.qvel)


def test_adaptation_sees_synchronized_physical_pose(raw):
    scales = {"link2": 1.2, "link4": 1.2}
    wrapped = ExpertAdaptationWrapper(SyncInitialPoseWrapper(raw, scales), scales)
    obs, info = wrapped.reset()
    assert info["sync_init_position_error_m"] < 1e-4
    assert info["exp_adap_ik_converged"]
    target_pos, target_rot = wrapped.nominal.pose(raw.nominal_obs["state.joint_position"])
    error = wrapped.nominal.pose_error(obs["state.joint_position"], target_pos, target_rot)
    assert np.linalg.norm(error[:3]) < 1e-4
    assert np.linalg.norm(error[3:]) < 1e-3


def test_unreachable_sync_fails_before_rebuild_and_never_uses_approximate_pose(raw, monkeypatch):
    wrapped = SyncInitialPoseWrapper(raw, {"link2": 1.2})
    monkeypatch.setattr(PandaKinematics, "inverse", lambda self, p, r, q: (q, np.ones(6)))
    with pytest.raises(RuntimeError, match="--sync_init could not match"):
        wrapped.reset()
    np.testing.assert_array_equal(raw.env.sim.data.qpos, raw.nominal_state.qpos)
    np.testing.assert_array_equal(raw.env.sim.data.qvel, raw.nominal_state.qvel)
    assert not wrapped._apply_scales
