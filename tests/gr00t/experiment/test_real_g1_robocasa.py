"""CPU regression tests for the REAL_G1 / RoboCasa adapter using a fake simulator.

Run with pytest. These check data conversion and rollout bookkeeping, not the
model's task success; no checkpoint, model server, or simulator assets are used.
"""

from contextlib import nullcontext
import json
from types import SimpleNamespace

from gr00t.experiment.custom import run_gr00t_on_robocas as experiment
from gr00t.experiment.custom.eef_target_viewer import (
    eef_positions_in_world,
    eef_rotations_in_world,
    rotation_distance_degrees,
)
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


def test_eef_marker_positions_use_absolute_pelvis_coordinates():
    base_rot = Rotation.from_euler("xyz", [0.3, -0.2, 1.1]).as_matrix()
    base_pos = np.array([2.0, -3.0, 0.8])
    wrist_pos = np.array([2.1, -3.2, 0.9])
    local_target = np.array([0.2, 0.3, 0.1])
    sim = SimpleNamespace(
        data=SimpleNamespace(
            get_body_xmat=lambda _: base_rot,
            get_body_xpos={"pelvis": base_pos, "wrist": wrist_pos}.__getitem__,
        )
    )
    command = np.r_[local_target, np.eye(3)[:2].ravel()]
    current, target = eef_positions_in_world(
        sim, "pelvis", {"left": "wrist"}, {"action.left_wrist_eef_9d": command}
    )["left"]
    np.testing.assert_allclose(current, wrist_pos)
    np.testing.assert_allclose(target, base_pos + base_rot @ local_target)
    # No second relative-action decode, and no mutation of the command/state.
    np.testing.assert_array_equal(command[:3], local_target)
    assert not np.shares_memory(current, wrist_pos)


def test_eef_rotation_decodes_rows_and_composes_pelvis_once():
    from gr00t.data.state_action.action_chunking import EndEffectorActionChunk
    from gr00t.data.state_action.pose import EndEffectorPose
    from gr00t.data.types import ActionFormat

    base_rot = Rotation.from_euler("xyz", [0.3, -0.2, 1.1]).as_matrix()
    current_rot = Rotation.from_euler("xyz", [-0.6, 0.5, 0.4]).as_matrix()
    delta_rot = Rotation.from_euler("xyz", [0.2, 0.1, -0.3]).as_matrix()
    # Exercise the same relative->absolute rotation composition as the policy
    # server, then ensure the viewer only applies pelvis->world afterward.
    reference = EndEffectorPose(rotation=current_rot, rotation_type="matrix")
    relative = EndEffectorActionChunk([EndEffectorPose(rotation=delta_rot, rotation_type="matrix")])
    command = relative.to_absolute_chunking(reference).to(ActionFormat.XYZ_ROT6D)[0]
    sim = SimpleNamespace(
        data=SimpleNamespace(
            get_body_xmat={"pelvis": base_rot, "wrist": base_rot @ current_rot}.__getitem__
        )
    )
    current, target = eef_rotations_in_world(
        sim, "pelvis", {"left": "wrist"}, {"action.left_wrist_eef_9d": command}
    )["left"]
    np.testing.assert_allclose(current, base_rot @ current_rot, atol=1e-7)
    np.testing.assert_allclose(target, base_rot @ current_rot @ delta_rot, atol=1e-7)
    expected_angle = np.degrees(Rotation.from_matrix(delta_rot).magnitude())
    assert rotation_distance_degrees(current, target) == pytest.approx(expected_angle)


@pytest.mark.parametrize("rotation", [[0.0] * 6, [1.0, 0, 0, 2.0, 0, 0]])
def test_degenerate_eef_rotation_has_no_invented_orientation(rotation):
    sim = SimpleNamespace(data=SimpleNamespace(get_body_xmat=lambda _: np.eye(3)))
    current, target = eef_rotations_in_world(
        sim, "pelvis", {"left": "wrist"}, {"action.left_wrist_eef_9d": np.r_[0, 0, 0, rotation]}
    )["left"]
    np.testing.assert_array_equal(current, np.eye(3))
    assert target is None


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


def test_eef_markers_follow_executed_steps_without_changing_commands_or_images(adapter):
    snapshots = []
    joint_snapshots = []

    def record(positions, joint_targets, rotations):
        snapshots.append(positions)
        joint_snapshots.append(joint_targets)

    # Vary the wrist position during each physics step to catch one-step-old
    # markers. Pelvis translation must apply to targets, not the wrist offset.
    base_pos = np.array([2.0, -3.0, 0.8])
    wrist_pos = np.zeros(3)
    adapter.env.sim.data.get_body_xpos = lambda body: (
        base_pos if body == adapter.base_body else wrist_pos
    )
    original_step = adapter.env.step

    def step(action):
        wrist_pos[:] += 0.1
        return original_step(action)

    adapter.env.step = step
    contract = experiment.PolicyHorizonSpec.from_modality_config(
        {
            "video": experiment.ModalityConfig(delta_indices=[0], modality_keys=["ego_view"]),
            "state": experiment.ModalityConfig(delta_indices=[0], modality_keys=list(JOINT_DIMS)),
            "action": experiment.ModalityConfig(
                delta_indices=list(range(4)), modality_keys=list(ACTION_DIMS)
            ),
        },
        n_action_steps=2,
    )
    env = experiment.MultiStepWrapper(adapter, contract=contract, max_episode_steps=2)
    adapter.env._check_success = lambda: False
    obs, _ = env.reset()
    initial_image = obs["video.ego_view"].copy()
    # Reset removes the previous episode's overlay; attach our recording sink.
    adapter.eef_viewer = SimpleNamespace(update=record)
    actions = {f"action.{g}": np.zeros((4, dim)) for g, dim in ACTION_DIMS.items()}
    for group in JOINT_DIMS:
        actions[f"action.{group}"][:] = 5.0  # The ghost must see clipped targets.
    for side, sign in (("left", 1), ("right", -1)):
        actions[f"action.{side}_wrist_eef_9d"][:, :3] = sign * np.arange(12).reshape(4, 3)
    result, *_ = env.step(actions)
    assert len(snapshots) == 2
    for index, snapshot in enumerate(snapshots):
        for side in ("left", "right"):
            current, target = snapshot[side]
            np.testing.assert_allclose(current, np.full(3, 0.1 * (index + 1)))
            np.testing.assert_allclose(
                target, base_pos + actions[f"action.{side}_wrist_eef_9d"][index, :3]
            )
    np.testing.assert_array_equal(result["video.ego_view"], initial_image)
    for target in adapter.env.last_action.values():
        np.testing.assert_array_equal(target, 2.0)
    for targets in joint_snapshots:
        for target in targets.values():
            np.testing.assert_array_equal(target, 2.0)
    for group in JOINT_DIMS:
        np.testing.assert_array_equal(actions[f"action.{group}"], 5.0)
    # Joint-only smoke actions clear previous policy targets, rather than
    # retaining stale markers or making up a zero EEF command.
    adapter.step({f"action.{g}": np.zeros(dim) for g, dim in JOINT_DIMS.items()})
    assert all(target is None for _, target in snapshots[-1].values())


def test_settling_holds_nonzero_joints_in_controller_and_hand_actuator_order(adapter):
    obs, _ = adapter.reset()
    adapter.step({f"action.{g}": obs[f"state.{g}"] for g in JOINT_DIMS})
    hold = make_joint_hold_action(adapter.env)
    for part, expected in adapter.env.last_action.items():
        np.testing.assert_allclose(hold[part], expected, rtol=1e-7)
    # Hand joint order in the XML is different from actuator order.
    np.testing.assert_allclose(hold["left_gripper"], obs["state.left_hand"][[4, 5, 6, 0, 1, 2, 3]])


def test_reject_wrong_policy_embodiment():
    with pytest.raises(ValueError, match="REAL_G1"):
        validate_modality_config({"video": SimpleNamespace(modality_keys=["front"])})


def test_rollout_counts_partial_chunk_and_does_not_double_count_reward(
    adapter, monkeypatch, capsys
):
    adapter.env._check_success = lambda: False
    monkeypatch.setattr(experiment, "create_env", lambda _: adapter)
    experiment.main(
        experiment.Config(
            smoke_test=True, n_action_steps=8, max_episode_steps=18, visualize=False, video_dir=None
        )
    )
    result = json.loads(capsys.readouterr().out.splitlines()[0])
    assert result["steps"] == 18
    assert result["reward"] == 18.0
    assert result["success"] is False


def test_input_viewer_matches_language_and_rgb_history_sent_to_policy(adapter, monkeypatch):
    from gr00t.policy import server_client

    snapshots = []
    requests = []
    viewer = SimpleNamespace(closed=False, polls=0)

    def update(instruction, frames, **metadata):
        snapshots.append((instruction, frames.copy(), metadata))

    def poll():
        viewer.polls += 1

    viewer.update = update
    viewer.poll = poll
    viewer.close = lambda: setattr(viewer, "closed", True)
    monkeypatch.setattr(experiment, "PolicyInputViewer", lambda *args: viewer)

    def create_env(config):
        adapter.instruction = config.instruction
        return adapter

    monkeypatch.setattr(experiment, "create_env", create_env)
    adapter.env._check_success = lambda: False
    original_step = adapter.env.step
    frame_count = 0

    def changing_camera(action):
        nonlocal frame_count
        frame_count += 1
        raw, reward, done, info = original_step(action)
        return {key: image + frame_count for key, image in raw.items()}, reward, done, info

    adapter.env.step = changing_camera
    modalities = {
        name: experiment.ModalityConfig(delta_indices=indices, modality_keys=keys)
        for name, indices, keys in (
            ("video", [-20, 0], ["ego_view"]),
            ("state", [0], list(experiment.STATE_DIMS)),
            ("action", list(range(40)), list(ACTION_DIMS)),
            ("language", [0], [LANGUAGE_KEY]),
        )
    }

    def get_action(obs):
        instruction, frames, metadata = snapshots[-1]
        assert instruction == obs[LANGUAGE_KEY][0] == "Put the apple on the plate."
        assert metadata["smoke_test"] is False
        np.testing.assert_array_equal(frames, obs["video.ego_view"][0])
        requests.append(metadata["step"])
        return {f"action.{key}": np.zeros((1, 40, dim)) for key, dim in ACTION_DIMS.items()}, {}

    policy = SimpleNamespace(
        get_modality_config=lambda: modalities, reset=lambda: None, get_action=get_action
    )
    monkeypatch.setattr(server_client, "PolicyClient", lambda **kwargs: nullcontext(policy))
    experiment.main(
        experiment.Config(instruction="Put the apple on the plate.", max_episode_steps=32)
    )
    assert requests == [0, 8, 16, 24]
    # At step 24 the history must contain step 4 and step 24, not two copies
    # of the current camera image. Frames retain their original RGB row order.
    first_rgb = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)[::-1]
    np.testing.assert_array_equal(snapshots[-1][1], np.stack([first_rgb + 4, first_rgb + 24]))
    assert viewer.polls == 32
    assert viewer.closed
