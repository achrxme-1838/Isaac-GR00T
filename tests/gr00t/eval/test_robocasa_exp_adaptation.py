"""Pose matching, observation consistency, and isolation from physical actions/state."""

from types import SimpleNamespace
import warnings

import gymnasium as gym
import numpy as np
import pytest
from scipy.spatial.transform import Rotation


mujoco = pytest.importorskip("mujoco")
pytest.importorskip("robosuite")

from gr00t.eval.sim.robocasa_exp_adaptation import (  # noqa: E402
    ExpertAdaptationWrapper,
    PandaKinematics,
    _rotation_log_jacobian,
)
from gr00t.eval.sim.robocasa_x_emb import scale_robot_xml  # noqa: E402
from robosuite.models.grippers.panda_gripper import PandaGripper  # noqa: E402
from robosuite.models.robots.compositional import PandaOmron  # noqa: E402


@pytest.fixture
def robot():
    panda = PandaOmron(idn=0)
    gripper = PandaGripper(idn="0_right")
    panda.add_gripper(gripper)
    return panda, gripper.important_sites["grip_site"]


def test_fk_matches_full_model_with_rotated_translated_base(robot):
    panda, site_name = robot
    xml = scale_robot_xml(panda.get_xml(), {"link0": 1.2, "link2": 1.2, "link4": 0.8})
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    model.body("robot0_base").pos[:] = [1.3, -2.0, 0.7]
    model.body("robot0_base").quat[:] = Rotation.from_euler("xyz", [0.2, -0.4, 1.1]).as_quat()[
        [3, 0, 1, 2]
    ]
    for i in range(7):
        data.joint(f"robot0_joint{i + 1}").qpos[:] = panda.init_qpos[i]
    mujoco.mj_forward(model, data)
    arm = PandaKinematics(xml, site_name)
    pos, rot = arm.pose(panda.init_qpos)
    root = data.body("robot0_link0")
    root_rotation = root.xmat.reshape(3, 3)
    np.testing.assert_allclose(
        pos, root_rotation.T @ (data.site(site_name).xpos - root.xpos), atol=1e-10
    )
    np.testing.assert_allclose(
        rot, root_rotation.T @ data.site(site_name).xmat.reshape(3, 3), atol=1e-10
    )


@pytest.mark.parametrize(
    "scales", [{"link2": 1.2, "link4": 1.2}, {"link0": 0.9}, {"link7": 1.3}, {"link2": 0.8}]
)
def test_changed_fk_to_nominal_ik_matches_position_and_orientation(robot, scales):
    panda, site_name = robot
    original = PandaKinematics(panda.get_xml(), site_name)
    changed = PandaKinematics(scale_robot_xml(panda.get_xml(), scales), site_name)
    q_real = panda.init_qpos + [0.1, 0.1, -0.1, 0.1, 0.2, 0.1, -0.1]
    target_pos, target_rot = changed.pose(q_real)
    q_nominal, error = original.inverse(target_pos, target_rot, q_real)
    assert np.linalg.norm(error[:3]) < 1e-4
    assert np.linalg.norm(error[3:]) < 1e-3
    assert np.all(q_nominal >= original.lower)
    assert np.all(q_nominal <= original.upper)
    assert not np.allclose(q_nominal, q_real)


def test_pose_residual_jacobian_matches_finite_differences(robot):
    panda, site_name = robot
    arm = PandaKinematics(panda.get_xml(), site_name)
    q = panda.init_qpos.copy()
    target_pos, target_rot = arm.pose(q + [0.2, -0.1, 0.3, -0.1, 0.2, 0.1, -0.2])
    analytic = arm.jacobian(q)
    rot_error = arm.pose_error(q, target_pos, target_rot)[3:]
    analytic[3:] = _rotation_log_jacobian(rot_error) @ analytic[3:]
    h = 1e-6
    numerical = np.column_stack(
        [
            (
                arm.pose_error(q + delta, target_pos, target_rot)
                - arm.pose_error(q - delta, target_pos, target_rot)
            )
            / (2 * h)
            for delta in np.eye(7) * h
        ]
    )
    np.testing.assert_allclose(analytic, numerical, atol=1e-7)


def test_unreachable_pose_keeps_joint_limits_and_reports_error(robot):
    panda, site_name = robot
    arm = PandaKinematics(panda.get_xml(), site_name)
    q, error = arm.inverse(np.array([10, 10, 10]), np.eye(3), panda.init_qpos)
    assert np.isfinite(q).all()
    assert np.all(q >= arm.lower) and np.all(q <= arm.upper)
    assert np.linalg.norm(error[:3]) > 1.0


class ArmObservationEnv(gym.Env):
    def __init__(self, robot):
        panda, site_name = robot
        self.env = SimpleNamespace(
            robots=[SimpleNamespace(robot_model=panda, eef_site_id={"right": 0})],
            sim=SimpleNamespace(model=SimpleNamespace(site_id2name=lambda site_id: site_name)),
        )
        q = panda.init_qpos.astype(np.float32)
        self.obs = {
            "state.joint_position": q,
            "state.joint_position_cos": np.cos(q),
            "state.joint_position_sin": np.sin(q),
            "state.joint_velocity": np.linspace(-0.03, 0.03, 7, dtype=np.float32),
            "state.end_effector_position_relative": np.zeros(3, dtype=np.float32),
            "video.test": np.zeros((8, 8, 3), dtype=np.uint8),
            "annotation.test": "open the drawer",
        }
        self.last_action = None

    def reset(self, **kwargs):
        return self.obs, {}

    def step(self, action):
        self.last_action = action
        return self.obs, 0.0, False, False, {"success": False}


def test_observation_and_velocity_mapping_preserves_other_fields_and_actions(robot):
    raw = ArmObservationEnv(robot)
    original_q = raw.obs["state.joint_position"].copy()
    env = ExpertAdaptationWrapper(raw, {"link2": 1.2, "link4": 1.2})
    obs, info = env.reset()
    assert info["exp_adap_ik_converged"]
    assert obs.keys() == raw.obs.keys()
    np.testing.assert_allclose(
        obs["state.joint_position_sin"], np.sin(obs["state.joint_position"]), atol=1e-7
    )
    np.testing.assert_allclose(
        obs["state.joint_position_cos"], np.cos(obs["state.joint_position"]), atol=1e-7
    )
    np.testing.assert_array_equal(raw.obs["state.joint_position"], original_q)
    for key, value in obs.items():
        if not key.startswith("state.joint_"):
            assert value is raw.obs[key]
        elif isinstance(value, np.ndarray):
            assert value.shape == (7,) and value.dtype == np.float32
    original_twist = env.modified.jacobian(original_q) @ raw.obs["state.joint_velocity"]
    adapted_twist = env.nominal.jacobian(obs["state.joint_position"]) @ obs["state.joint_velocity"]
    np.testing.assert_allclose(adapted_twist, original_twist, atol=1e-5)
    action = {"action.end_effector_position": np.array([0.1, 0, 0])}
    env.step(action)
    assert raw.last_action is action

    old_kinematics = env.nominal
    env.previous_q = np.zeros(7)
    new_obs, _ = env.reset()
    assert env.nominal is not old_kinematics
    np.testing.assert_allclose(new_obs["state.joint_position"], obs["state.joint_position"])


@pytest.mark.parametrize("scales", [{}, {"link2": 1.0, "link4": 1.0}])
def test_unchanged_embodiment_is_exact_identity(robot, scales):
    raw = ArmObservationEnv(robot)
    env = ExpertAdaptationWrapper(raw, scales)
    obs, info = env.reset()
    assert obs is raw.obs
    assert info == {}


def test_nonconvergence_warning_is_once_per_episode(robot):
    env = ExpertAdaptationWrapper(ArmObservationEnv(robot), {"link2": 10})
    with pytest.warns(UserWarning, match="outside tolerance"):
        _, info = env.reset()
    assert not info["exp_adap_ik_converged"]
    with warnings.catch_warnings(record=True) as caught:
        env.step({})
    assert not caught


def test_diagnostics_count_reset_and_every_step_and_restart_each_episode(robot, monkeypatch):
    env = ExpertAdaptationWrapper(ArmObservationEnv(robot), {"link2": 1.2})
    _, initial = env.reset()
    assert initial["exp_adap_ik_samples"] == 1
    assert initial["exp_adap_ik_failures"] == 0
    errors = iter(
        [np.array([0.01, 0, 0, 0, 0, 0.02]), np.zeros(6), np.array([0.002, 0, 0, 0, 0, 0.04])]
    )
    monkeypatch.setattr(env.nominal, "inverse", lambda p, r, seed: (seed.copy(), next(errors)))
    with pytest.warns(UserWarning, match="outside tolerance"):
        env.step({})
    env.step({})
    _, _, _, _, info = env.step({})
    assert info["exp_adap_ik_samples"] == 4
    assert info["exp_adap_ik_failures"] == 2
    assert info["exp_adap_ik_failure_rate"] == 0.5
    assert info["exp_adap_ik_max_position_error_m"] == 0.01
    assert info["exp_adap_ik_max_rotation_error_rad"] == 0.04
    _, reset_info = env.reset()
    assert reset_info["exp_adap_ik_samples"] == 1
    assert reset_info["exp_adap_ik_failures"] == 0
    assert reset_info["exp_adap_ik_max_position_error_m"] < 1e-4
