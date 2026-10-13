"""Physical reset perturbations and per-control-step velocity suppression, without GL."""

from types import SimpleNamespace

from gr00t.eval._horizon_contract import PolicyHorizonSpec
from gr00t.eval.sim.wrapper.multistep_wrapper import MultiStepWrapper
import gymnasium as gym
import numpy as np
import pytest
from scipy.spatial.transform import Rotation


mujoco = pytest.importorskip("mujoco")
pytest.importorskip("robosuite")

from gr00t.eval.sim.robocasa_exp_adaptation import ARM_JOINTS, PandaKinematics  # noqa: E402
from gr00t.eval.sim.robocasa_perturbations import PandaPerturbationWrapper  # noqa: E402
from robosuite.models.bases import OmronMobileBase  # noqa: E402
from robosuite.models.grippers.panda_gripper import PandaGripper  # noqa: E402
from robosuite.models.robots.compositional import PandaOmron  # noqa: E402
from robosuite.utils.binding_utils import MjSim  # noqa: E402
from scripts.eval_robocasa_gui import parse_args  # noqa: E402


class ArmEnv(gym.Env):
    """Real Panda FK/state with a small Gym adapter; no kitchen assets or renderer."""

    def __init__(self):
        panda = PandaOmron(idn=0)
        panda.add_base(OmronMobileBase(idn=0))
        gripper = PandaGripper(idn="0_right")
        panda.add_gripper(gripper)
        sim = MjSim.from_xml_string(panda.get_xml())
        self.model = sim.model._model
        self.data = sim.data._data
        joint_ids = [self.model.joint(name).id for name in ARM_JOINTS]
        self.qpos_indices = self.model.jnt_qposadr[joint_ids]
        self.dof_indices = self.model.jnt_dofadr[joint_ids]
        self.other_qpos = np.setdiff1d(np.arange(self.model.nq), self.qpos_indices)
        self.other_dofs = np.setdiff1d(np.arange(self.model.nv), self.dof_indices)
        self.site_id = self.model.site(gripper.important_sites["grip_site"]).id
        self.default_q = panda.init_qpos.copy()
        self.default_qpos = self.model.qpos0.copy()
        self.default_qpos[self.qpos_indices] = self.default_q
        self.default_qpos[self.other_qpos] = 0.02
        self.default_qpos[self.model.joint("mobilebase0_joint_mobile_yaw").qposadr] = 0.8
        self.default_qvel = np.linspace(0.01, 0.1, self.model.nv)
        self.controller = SimpleNamespace(
            update_initial_joints=lambda q: setattr(self, "controller_q", q.copy()),
            reset_goal=lambda: setattr(
                self, "controller_goal", self.data.site_xpos[self.site_id].copy()
            ),
        )
        self.env = SimpleNamespace(
            sim=sim,
            robots=[
                SimpleNamespace(
                    robot_model=panda,
                    eef_site_id={"right": self.site_id},
                    part_controllers={"right": self.controller},
                )
            ],
            viewer_get_obs=False,
            _obs_cache={},
            _get_observations=self.observation,
        )
        obs, _ = self.reset()
        self.observation_space = gym.spaces.Dict(
            {
                key: gym.spaces.Box(-np.inf, np.inf, shape=value.shape, dtype=value.dtype)
                for key, value in obs.items()
            }
        )
        self.action_space = gym.spaces.Dict({"action.test": gym.spaces.Box(-1, 1, (7,))})

    def observation(self, *, force_update=False):
        q = self.data.qpos[self.qpos_indices].astype(np.float32)
        return {
            "state.joint_position": q,
            "state.joint_position_cos": np.cos(q),
            "state.joint_position_sin": np.sin(q),
            "state.joint_velocity": self.data.qvel[self.dof_indices].astype(np.float32),
            "state.ee_pos": self.data.site_xpos[self.site_id].astype(np.float32).copy(),
            # A synthetic camera reading checks that reset refreshes images too.
            "video.test": q.reshape(1, 7, 1).copy(),
        }

    def get_basic_observation(self, obs):
        return obs

    def get_groot_observation(self, obs):
        return obs

    def reset(self, **kwargs):
        self.data.qpos[:] = self.default_qpos
        self.data.qvel[:] = self.default_qvel
        self.data.time = 0
        self.env.sim.forward()
        self.step_inputs = []
        self.raw_obs = self.observation()
        self.info = {"success": False}
        return self.raw_obs, self.info

    def step(self, action):
        self.step_inputs.append(self.data.qvel[self.dof_indices].copy())
        self.last_action = action
        self.data.qvel[self.dof_indices] = 0.3
        self.data.qpos[self.qpos_indices] += 0.001
        self.env.sim.forward()
        return self.observation(), 0.5, False, False, {"success": False}


@pytest.fixture
def raw():
    return ArmEnv()


def test_cli_defaults_and_custom_magnitudes():
    args = parse_args([])
    assert (args.initial_ee_pos, args.random_joint, args.no_vel) == (0, 0, False)
    args = parse_args(["--initial_ee_pos", "--random_joint", "--no_vel"])
    assert (args.initial_ee_pos, args.random_joint, args.no_vel) == (0.05, 0.1, True)
    args = parse_args(["--initial-ee-pos=0.02", "--random-joint=0.04", "--no-vel"])
    assert (args.initial_ee_pos, args.random_joint, args.no_vel) == (0.02, 0.04, True)


@pytest.mark.parametrize("flag", ["--initial_ee_pos", "--random_joint"])
@pytest.mark.parametrize("value", ["-0.1", "nan", "inf", "nope"])
def test_invalid_magnitudes_fail_before_launch(flag, value):
    with pytest.raises(SystemExit):
        parse_args([f"{flag}={value}"])


def test_disabled_is_identity(raw):
    wrapped = PandaPerturbationWrapper(raw)
    obs, info = wrapped.reset(seed=2)
    assert obs is raw.raw_obs and info is raw.info
    np.testing.assert_array_equal(raw.data.qpos, raw.default_qpos)
    np.testing.assert_array_equal(raw.data.qvel, raw.default_qvel)
    wrapped.step({})
    np.testing.assert_array_equal(raw.data.qvel[raw.dof_indices], 0.3)


def test_initial_ee_matches_target_preserves_orientation_and_refreshes_observations(raw):
    original_position = raw.data.site_xpos[raw.site_id].copy()
    original_rotation = raw.data.site_xmat[raw.site_id].reshape(3, 3).copy()
    wrapped = PandaPerturbationWrapper(raw, initial_ee_pos=0.05)
    obs, info = wrapped.reset(seed=7)
    offset = info["perturbation_ee_offset_m"]
    assert np.linalg.norm(offset) == pytest.approx(0.05)
    assert np.linalg.norm(raw.data.site_xpos[raw.site_id] - original_position) == pytest.approx(
        0.05, abs=1e-4
    )
    np.testing.assert_allclose(
        raw.data.site_xpos[raw.site_id], original_position + offset, atol=1e-4
    )
    rotation = raw.data.site_xmat[raw.site_id].reshape(3, 3)
    assert np.linalg.norm(Rotation.from_matrix(rotation @ original_rotation.T).as_rotvec()) < 1e-3
    np.testing.assert_allclose(obs["state.ee_pos"], wrapped.target, atol=1e-4)
    np.testing.assert_allclose(obs["state.joint_position"], raw.data.qpos[raw.qpos_indices])
    np.testing.assert_allclose(obs["state.joint_position_cos"], np.cos(obs["state.joint_position"]))
    np.testing.assert_allclose(obs["state.joint_position_sin"], np.sin(obs["state.joint_position"]))
    np.testing.assert_array_equal(obs["video.test"].ravel(), obs["state.joint_position"])
    np.testing.assert_array_equal(raw.data.qpos[raw.other_qpos], raw.default_qpos[raw.other_qpos])
    np.testing.assert_array_equal(raw.data.qvel[raw.other_dofs], raw.default_qvel[raw.other_dofs])
    np.testing.assert_allclose(raw.controller_q, raw.data.qpos[raw.qpos_indices])
    np.testing.assert_allclose(raw.controller_goal, raw.data.site_xpos[raw.site_id])
    assert raw.data.time == 0


@pytest.mark.parametrize("ee_radius", [0, 0.05])
def test_joint_noise_reproducibility_and_reset_reapplication(raw, ee_radius):
    wrapped = PandaPerturbationWrapper(raw, initial_ee_pos=ee_radius, random_joint=0.1)
    first, first_info = wrapped.reset(seed=17)
    delta = first_info["perturbation_joint_offset_rad"]
    assert np.all(np.abs(delta) <= 0.1) and np.any(delta > 0) and np.any(delta < 0)
    wrapped.step({})
    # Without no_vel, velocities after reset remain unconstrained.
    np.testing.assert_array_equal(raw.data.qvel[raw.dof_indices], 0.3)
    repeated, repeated_info = wrapped.reset(seed=17)
    np.testing.assert_array_equal(first["state.joint_position"], repeated["state.joint_position"])
    for key in first_info:
        np.testing.assert_array_equal(first_info[key], repeated_info[key])
    different, _ = wrapped.reset(seed=18)
    assert not np.array_equal(first["state.joint_position"], different["state.joint_position"])


def test_joint_noise_clips_to_joint_limits(raw):
    wrapped = PandaPerturbationWrapper(raw, random_joint=20)
    wrapped.reset(seed=4)
    ids = [raw.model.joint(name).id for name in ARM_JOINTS]
    lower, upper = raw.model.jnt_range[ids].T
    q = raw.data.qpos[raw.qpos_indices]
    assert np.all(q >= lower) and np.all(q <= upper)
    assert np.any(q == lower) or np.any(q == upper)


def test_no_vel_applies_to_every_step_in_action_chunk_and_preserves_other_dofs(raw):
    perturbations = PandaPerturbationWrapper(raw, no_vel=True)
    wrapped = MultiStepWrapper(
        perturbations,
        contract=PolicyHorizonSpec(3, 3, (0,), (-2, -1, 0)),
    )
    obs, _ = wrapped.reset(seed=0)
    np.testing.assert_array_equal(obs["state.joint_velocity"], 0)
    action = {"action.test": np.ones((3, 7))}
    obs, reward, terminated, truncated, info = wrapped.step(action)
    assert info["n_env_steps"] == 3 and len(raw.step_inputs) == 3
    assert reward == 0.5 and not terminated and not truncated
    np.testing.assert_array_equal(raw.step_inputs, np.zeros((3, 7)))
    np.testing.assert_array_equal(obs["state.joint_velocity"], np.zeros((3, 7)))
    np.testing.assert_array_equal(raw.last_action["action.test"], action["action.test"][-1])
    np.testing.assert_array_equal(raw.data.qvel[raw.dof_indices], 0)
    np.testing.assert_array_equal(raw.data.qvel[raw.other_dofs], raw.default_qvel[raw.other_dofs])
    np.testing.assert_array_equal(raw.data.qpos[raw.other_qpos], raw.default_qpos[raw.other_qpos])


def test_unreachable_ee_reports_failure_without_applying_approximation(raw, monkeypatch):
    monkeypatch.setattr(PandaKinematics, "inverse", lambda self, p, r, q: (q, np.ones(6)))
    wrapped = PandaPerturbationWrapper(raw, initial_ee_pos=0.05)
    with pytest.raises(RuntimeError, match="after 20 samples"):
        wrapped.reset(seed=0)
    np.testing.assert_array_equal(raw.data.qpos, raw.default_qpos)


def test_sphere_overlay_preserves_physics_and_appends_to_recording_scene(raw):
    wrapped = PandaPerturbationWrapper(raw, initial_ee_pos=0.05)
    obs, _ = wrapped.reset(seed=0)
    original = raw.data.qpos.copy(), raw.data.qvel.copy()
    original_ngeom = raw.model.ngeom
    overlay = wrapped.make_overlay(raw.env)
    scene = mujoco.MjvScene(raw.model, maxgeom=100)
    for start in (0, 3):
        overlay.update(scene, obs["state.joint_position"], start=start)
        assert scene.ngeom == start + 2
        np.testing.assert_allclose(scene.geoms[start].pos, wrapped.center, atol=1e-6)
        np.testing.assert_allclose(scene.geoms[start].size, 0.05)
        np.testing.assert_allclose(scene.geoms[start + 1].pos, wrapped.target, atol=1e-6)
        assert scene.geoms[start].rgba[3] < 0.2
    np.testing.assert_array_equal(raw.data.qpos, original[0])
    np.testing.assert_array_equal(raw.data.qvel, original[1])
    assert raw.model.ngeom == original_ngeom
