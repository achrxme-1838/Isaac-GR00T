"""CPU regression tests for the REAL_G1 / RoboCasa adapter using a fake simulator.

Run with pytest. These check data conversion and rollout bookkeeping, not the
model's task success; no checkpoint, model server, or simulator assets are used.
"""

import json
from types import SimpleNamespace

from gr00t.experiment.custom import run_gr00t_on_robocas as experiment
from gr00t.experiment.custom.run_gr00t_on_robocas import (
    ACTION_DIMS,
    JOINT_DIMS,
    LANGUAGE_KEY,
    RealG1TabletopEnv,
    make_joint_hold_action,
    validate_modality_config,
    wrist_eef_in_base,
)
import numpy as np
import pytest
from scipy.spatial.transform import Rotation


def test_wrist_pose_uses_pelvis_frame_and_rotation_rows():
    base_rot = Rotation.from_euler("xyz", [0.3, -0.2, 1.1]).as_matrix()
    local_rot = Rotation.from_euler("xyz", [-0.6, 0.5, 0.4]).as_matrix()
    base_pos = np.array([2.0, -3.0, 0.8])
    local_pos = np.array([0.2, 0.3, 0.1])
    sim = SimpleNamespace(
        data=SimpleNamespace(
            get_body_xmat={"base": base_rot, "wrist": base_rot @ local_rot}.__getitem__,
            get_body_xpos={
                "base": base_pos,
                "wrist": base_pos + base_rot @ local_pos,
            }.__getitem__,
        )
    )
    result = wrist_eef_in_base(sim, "base", "wrist")
    np.testing.assert_allclose(result[:3], local_pos, atol=1e-7)
    np.testing.assert_allclose(result[3:], local_rot[:2].ravel(), atol=1e-7)
    assert result.dtype == np.float32


@pytest.fixture
def adapter():
    # Controller order deliberately differs from policy order, including the
    # thumb-first hand XML versus index/middle/thumb policy convention.
    group_parts = {
        "left_arm": "left",
        "right_arm": "right",
        "left_hand": "left_gripper",
        "right_hand": "right_gripper",
        "waist": "torso",
    }
    groups, names, controllers, actuator_indexes = {}, [], {}, {}
    for group, dim in JOINT_DIMS.items():
        joints = [f"{group}_{i}" for i in range(dim)]
        groups[group] = {"joints": joints}
        prefix = f"gripper_{group.split('_')[0]}_" if group.endswith("hand") else "robot0_"
        indexes = list(range(len(names), len(names) + dim))
        names.extend(prefix + name for name in joints)
        part = group_parts[group]
        order = [4, 5, 6, 0, 1, 2, 3] if group.endswith("hand") else list(reversed(range(dim)))
        indexes = [indexes[i] for i in order]
        controllers[part] = SimpleNamespace(input_type="absolute", joint_index=indexes)
        actuator_indexes[part] = indexes
    model = SimpleNamespace(
        get_joint_qpos_addr=names.index,
        joint_name2id=names.index,
        joint_id2name=names.__getitem__,
        actuator_trnid=np.column_stack([np.arange(len(names)), -np.ones(len(names))]).astype(int),
        jnt_range=np.tile([-2.0, 2.0], (len(names), 1)),
    )
    robot = SimpleNamespace(
        robot_model=SimpleNamespace(naming_prefix="robot0_"),
        gripper={s: SimpleNamespace(naming_prefix=f"gripper_{s}_") for s in ("left", "right")},
        composite_controller=SimpleNamespace(part_controllers=controllers),
        _ref_actuators_indexes_dict=actuator_indexes,
        create_action_vector=lambda parts: parts,
    )
    sim = SimpleNamespace(
        model=model,
        data=SimpleNamespace(
            qpos=np.linspace(0.1, 1.1, len(names)),
            get_body_xmat=lambda _: np.eye(3),
            get_body_xpos=lambda _: np.zeros(3),
        ),
    )
    image = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
    raw = {"robot0_oak_egoview_image": image}
    env = SimpleNamespace(
        robots=[robot],
        sim=sim,
        camera_heights=[2],
        camera_widths=[3],
        get_ep_meta=lambda: {"lang": "Pick up the apple."},
        reset=lambda: raw,
        _check_success=lambda: True,
        close=lambda: None,
    )

    def step(action):
        env.last_action = action
        return raw, 1.0, False, {}

    env.step = step
    info = SimpleNamespace(
        root_frame_name="pelvis",
        hand_frame_names={"left": "left_wrist", "right": "right_wrist"},
        joint_groups=groups,
    )
    return RealG1TabletopEnv(env, info)


def test_joint_observations_and_absolute_commands_follow_names(adapter):
    obs, _ = adapter.reset(seed=5)
    np.testing.assert_array_equal(obs["video.ego_view"][0], np.arange(9, 18).reshape(3, 3))
    assert obs[LANGUAGE_KEY] == "Pick up the apple."
    actions = {}
    for group in JOINT_DIMS:
        expected = adapter.env.sim.data.qpos[adapter.qpos_indices[group]]
        np.testing.assert_allclose(obs[f"state.{group}"], expected)
        actions[f"action.{group}"] = expected.copy()
    # Redundant EEF and mobile outputs must not reach the joint controllers.
    actions.update(
        {f"action.{k}": np.full(d, 100.0) for k, d in ACTION_DIMS.items() if k not in JOINT_DIMS}
    )
    _, reward, _, _, info = adapter.step(actions)
    assert reward == 1.0 and info["success"]
    for part, (group, order) in adapter.part_joint_names.items():
        np.testing.assert_allclose(adapter.env.last_action[part], actions[f"action.{group}"][order])
    np.testing.assert_allclose(
        adapter.env.last_action["left_gripper"], actions["action.left_hand"][[4, 5, 6, 0, 1, 2, 3]]
    )


@pytest.mark.parametrize("bad", [np.zeros(6), np.full(7, np.nan), np.full(7, np.inf)])
def test_reject_invalid_joint_commands(adapter, bad):
    actions = {f"action.{g}": np.zeros(d) for g, d in JOINT_DIMS.items()}
    actions["action.left_arm"] = bad
    with pytest.raises(ValueError, match="Invalid left_arm"):
        adapter.step(actions)
    assert not hasattr(adapter.env, "last_action")


def test_joint_commands_are_clipped_in_radians(adapter):
    adapter.step({f"action.{g}": np.full(d, 5.0) for g, d in JOINT_DIMS.items()})
    for target in adapter.env.last_action.values():
        np.testing.assert_array_equal(target, 2.0)


def test_settling_holds_nonzero_joints_in_controller_and_hand_actuator_order(adapter):
    obs, _ = adapter.reset()
    adapter.step({f"action.{g}": obs[f"state.{g}"] for g in JOINT_DIMS})
    hold = make_joint_hold_action(adapter.env)
    for part, expected in adapter.env.last_action.items():
        np.testing.assert_allclose(hold[part], expected, rtol=1e-7)
    # Hand joint order in the XML is different from actuator order.
    np.testing.assert_allclose(
        hold["left_gripper"], obs["state.left_hand"][[4, 5, 6, 0, 1, 2, 3]]
    )


def test_reject_wrong_policy_embodiment():
    with pytest.raises(ValueError, match="REAL_G1"):
        validate_modality_config({"video": SimpleNamespace(modality_keys=["front"])})


def test_rollout_counts_partial_chunk_and_does_not_double_count_reward(
    adapter, monkeypatch, capsys
):
    adapter.env._check_success = lambda: False
    monkeypatch.setattr(experiment, "create_env", lambda _: adapter)
    experiment.main(
        experiment.Config(smoke_test=True, n_action_steps=8, max_episode_steps=18, video_dir=None)
    )
    result = json.loads(capsys.readouterr().out.splitlines()[0])
    assert result["steps"] == 18
    assert result["reward"] == 18.0
    assert result["success"] is False
