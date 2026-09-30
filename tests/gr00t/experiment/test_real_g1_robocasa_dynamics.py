"""Optional real simulator regressions; no policy server or checkpoint required.

Requires the RoboCasa tabletop assets, WBC checkout, and offscreen rendering.
Run with GR00T_RUN_G1_SIM_TESTS=1 python -m pytest <this file>.
"""

import os

from gr00t.experiment.custom import run_gr00t_on_robocas as experiment
import numpy as np
import pytest


pytestmark = pytest.mark.skipif(
    os.environ.get("GR00T_RUN_G1_SIM_TESTS") != "1",
    reason="Set GR00T_RUN_G1_SIM_TESTS=1 with the G1/RoboCasa simulator assets installed",
)


@pytest.fixture(scope="module")
def simulator():
    os.environ.setdefault("MUJOCO_GL", "egl")
    env = experiment.create_env(experiment.Config(visualize=False))
    try:
        yield env
    finally:
        env.close()


@pytest.mark.parametrize(
    "group,index,delta", [("left_arm", 1, 0.1), ("right_arm", 1, -0.1), ("waist", 0, 0.05)]
)
def test_small_joint_step_settles_without_exciting_other_limbs(simulator, group, index, delta):
    obs, _ = simulator.reset(seed=0)
    targets = {f"action.{key}": obs[f"state.{key}"].copy() for key in experiment.JOINT_DIMS}
    targets[f"action.{group}"][index] += delta
    for _ in range(40):
        obs, *_ = simulator.step(targets)

    # With GR1's kd=200 this small step causes persistent oscillation, even
    # in the uncommanded arm: errors exceed 0.1 rad and speeds exceed 1 rad/s.
    for key in ("left_arm", "right_arm", "waist"):
        np.testing.assert_allclose(obs[f"state.{key}"], targets[f"action.{key}"], atol=0.01)
    robot = simulator.env.robots[0]
    for part in ("left", "right", "torso"):
        controller = robot.composite_controller.part_controllers[part]
        velocity = simulator.env.sim.data.qvel[controller.qvel_index]
        assert np.max(np.abs(velocity)) < 0.05, (part, velocity)


def test_ghost_matches_joint_target_fk_without_changing_simulator_or_camera(simulator):
    from gr00t.experiment.custom.joint_target_viewer import JointTargetGhost
    import mujoco

    obs, _ = simulator.reset(seed=0)
    sim = simulator.env.sim
    model, live = sim.model._model, sim.data._data
    before_qpos, before_qvel, before_ctrl = live.qpos.copy(), live.qvel.copy(), live.ctrl.copy()
    before_rgba = model.geom_rgba.copy()
    before_image = sim.render(width=640, height=480, camera_name=simulator.camera_name).copy()
    ghost = JointTargetGhost(sim, simulator.joint_names, simulator.qpos_indices)
    targets = {g: obs[f"state.{g}"].copy() for g in experiment.JOINT_DIMS}
    targets["left_arm"][0] -= 0.4
    targets["right_arm"][1] -= 0.3
    targets["waist"][0] += 0.15
    targets["left_hand"][0] += 0.25
    ghost.update(targets)
    scene = mujoco.MjvScene(model, maxgeom=1000)
    ghost.append_to(scene)
    assert scene.ngeom == len(ghost.geom_sides) > 0
    assert set(ghost.geom_sides.values()) == {"left", "right"}

    # Compare isolated kinematics with a full MuJoCo forward pass at the
    # commanded joints, including waist and hand targets and actual root pose.
    reference = mujoco.MjData(model)
    reference.qpos[:] = before_qpos
    for group, target in targets.items():
        reference.qpos[simulator.qpos_indices[group]] = target
    mujoco.mj_forward(model, reference)
    drawn_ids = [
        g.objid
        for g in ghost.scene.geoms[: ghost.scene.ngeom]
        if g.objtype == mujoco.mjtObj.mjOBJ_GEOM and g.objid in ghost.geom_sides
    ]
    for rendered, geom_id in zip(scene.geoms[: scene.ngeom], drawn_ids, strict=True):
        np.testing.assert_allclose(rendered.pos, reference.geom_xpos[geom_id], atol=1e-6)
        np.testing.assert_allclose(rendered.mat.ravel(), reference.geom_xmat[geom_id], atol=1e-6)
        assert 0 < rendered.rgba[3] < 1
        assert rendered.transparent and rendered.matid == -1
    for side in ("left", "right"):
        wrist = model.body(simulator.wrist_bodies[side]).id
        assert np.linalg.norm(ghost.data.xpos[wrist] - live.xpos[wrist]) > 0.01
        wrist_pos, wrist_rot = ghost.wrist_pose(side)
        np.testing.assert_allclose(wrist_pos, reference.xpos[wrist], atol=1e-7)
        np.testing.assert_allclose(wrist_rot, reference.xmat[wrist].reshape(3, 3), atol=1e-7)
    np.testing.assert_array_equal(live.qpos, before_qpos)
    np.testing.assert_array_equal(live.qvel, before_qvel)
    np.testing.assert_array_equal(live.ctrl, before_ctrl)
    np.testing.assert_array_equal(model.geom_rgba, before_rgba)
    np.testing.assert_array_equal(
        sim.render(width=640, height=480, camera_name=simulator.camera_name), before_image
    )
