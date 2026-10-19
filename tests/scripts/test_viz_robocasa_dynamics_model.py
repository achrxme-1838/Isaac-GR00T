"""Check state replay and EEF-only decoration against independent MuJoCo poses."""

from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation


mujoco = pytest.importorskip("mujoco")
pytest.importorskip("robosuite")

from robosuite.models.bases import OmronMobileBase  # noqa: E402
from robosuite.models.grippers.panda_gripper import PandaGripper  # noqa: E402
from robosuite.models.robots.compositional import PandaOmron  # noqa: E402
from scripts.visualizer.viz_robocasa_dynamics_model import EefGhost, RobotStateReplay  # noqa: E402


@pytest.fixture
def robot_env():
    robot = PandaOmron(idn=0)
    robot.add_base(OmronMobileBase(idn=0))
    robot.add_gripper(PandaGripper(idn="0_right"))
    model = mujoco.MjModel.from_xml_string(robot.get_xml())
    data = mujoco.MjData(model)
    # A translated/rotated root mimics RoboCasa's randomized kitchen placement.
    root = model.body("robot0_base").id
    model.body_pos[root] = [1.1, -0.4, 0.05]
    model.body_quat[root] = Rotation.from_euler("z", 0.65).as_quat()[[3, 0, 1, 2]]
    for i, value in enumerate(robot.init_qpos, 1):
        data.joint(f"robot0_joint{i}").qpos[:] = value
    for name, value in {
        "mobile_forward": 0.4,
        "mobile_side": -0.2,
        "mobile_yaw": -1.0,
        "torso_height": 0.16,
    }.items():
        data.joint(f"mobilebase0_joint_{name}").qpos[:] = value
    data.joint("gripper0_right_finger_joint1").qpos[:] = 0.025
    data.joint("gripper0_right_finger_joint2").qpos[:] = -0.025
    mujoco.mj_forward(model, data)
    return SimpleNamespace(
        sim=SimpleNamespace(model=SimpleNamespace(_model=model), data=SimpleNamespace(_data=data))
    )


def observed_state(model, data):
    values = {
        "joint_pos": np.array([data.joint(f"robot0_joint{i}").qpos[0] for i in range(1, 8)]),
        "gripper_pos": np.array(
            [data.joint(f"gripper0_right_finger_joint{i}").qpos[0] for i in (1, 2)]
        ),
        "base_pos_w": data.site("mobilebase0_center").xpos.copy(),
        "base_quat_w": Rotation.from_matrix(
            data.site("mobilebase0_center").xmat.reshape(3, 3)
        ).as_quat(),
        "ee_pos_w": data.site("gripper0_right_grip_site").xpos.copy(),
        "ee_quat_w": Rotation.from_matrix(
            data.body("robot0_right_hand").xmat.reshape(3, 3)
        ).as_quat(),
    }
    mapping, offset = {}, 0
    for name, value in values.items():
        mapping[name] = SimpleNamespace(start=offset, end=offset + len(value))
        offset += len(value)
    return np.concatenate(list(values.values())), mapping


def test_replay_recovers_arm_fingers_and_mobile_base_without_physics(robot_env, monkeypatch):
    m, d = robot_env.sim.model._model, robot_env.sim.data._data
    state, mapping = observed_state(m, d)
    expected = d.qpos.copy()
    d.qpos[:] = 0
    d.qvel[:] = 3
    d.ctrl[:] = 0.5
    monkeypatch.setattr(mujoco, "mj_step", lambda *a: pytest.fail("Replay must never step physics"))
    replay = RobotStateReplay(robot_env, mapping)
    errors = replay.set_state(state, 3.5)
    np.testing.assert_allclose(d.qpos, expected, atol=1e-10)
    np.testing.assert_array_equal(d.qvel, 0)
    np.testing.assert_array_equal(d.ctrl, 0)
    assert d.time == 3.5
    assert max(errors) < 1e-10
    # Repeated application must not drift or depend on the preceding pose.
    replay.set_state(state, 3.55)
    np.testing.assert_allclose(d.qpos, expected, atol=1e-10)


def test_ghost_matches_eef_at_another_pose_without_moving_live_robot(robot_env):
    m, d = robot_env.sim.model._model, robot_env.sim.data._data
    reference = mujoco.MjData(m)
    reference.qpos[:] = d.qpos
    reference.joint("robot0_joint2").qpos[:] += 0.4
    reference.joint("robot0_joint5").qpos[:] -= 0.3
    mujoco.mj_forward(m, reference)
    pose = np.r_[
        reference.site("gripper0_right_grip_site").xpos,
        Rotation.from_matrix(reference.body("robot0_right_hand").xmat.reshape(3, 3)).as_quat(),
    ]
    before = d.qpos.copy(), d.qvel.copy(), d.geom_xpos.copy(), m.ngeom
    ghost = EefGhost(robot_env, (0.1, 1.0, 0.1, 0.35))
    scene = mujoco.MjvScene(m, maxgeom=500)
    ghost.update(scene, pose, start=2)
    assert scene.ngeom == len(ghost.geom_ids) + 2
    for index, geom_id in enumerate(ghost.geom_ids, 2):
        rendered = scene.geoms[index]
        assert m.body(m.geom_bodyid[geom_id]).name.startswith(("gripper", "robot0_right_hand"))
        np.testing.assert_allclose(rendered.pos, reference.geom_xpos[geom_id], atol=1e-6)
        np.testing.assert_allclose(rendered.mat.ravel(), reference.geom_xmat[geom_id], atol=1e-6)
        np.testing.assert_allclose(rendered.rgba, [0.1, 1, 0.1, 0.35])
        assert rendered.matid == -1
        assert rendered.objtype == mujoco.mjtObj.mjOBJ_UNKNOWN
        if rendered.type == mujoco.mjtGeom.mjGEOM_MESH:
            assert rendered.dataid == 2 * m.geom_dataid[geom_id]
    for actual, expected in zip((d.qpos, d.qvel, d.geom_xpos, m.ngeom), before):
        np.testing.assert_array_equal(actual, expected)
    # An unreachable prediction must stay at its predicted location, not be IK-clipped.
    pose[0] += 10
    pose[3:] *= -1
    ghost.update(scene, pose)
    for index, geom_id in enumerate(ghost.geom_ids):
        np.testing.assert_allclose(
            scene.geoms[index].pos, reference.geom_xpos[geom_id] + [10, 0, 0], atol=1e-6
        )


def test_replay_rejects_inconsistent_observation_frames(robot_env):
    m, d = robot_env.sim.model._model, robot_env.sim.data._data
    state, mapping = observed_state(m, d)
    state[mapping["ee_pos_w"].start] += 0.01
    with pytest.raises(ValueError, match="convention mismatch"):
        RobotStateReplay(robot_env, mapping).set_state(state, 0)
