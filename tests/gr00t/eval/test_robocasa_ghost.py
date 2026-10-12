"""The ghost follows virtual joints/world base without changing physics or sensor geoms."""

from contextlib import contextmanager
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest


mujoco = pytest.importorskip("mujoco")
pytest.importorskip("robosuite")

from gr00t.eval.sim.robocasa_ghost import NominalArmGhost  # noqa: E402
from gr00t.eval.sim.robocasa_x_emb import scale_robot_xml  # noqa: E402
from robosuite.models.bases import OmronMobileBase  # noqa: E402
from robosuite.models.grippers.panda_gripper import PandaGripper  # noqa: E402
from robosuite.models.robots.compositional import PandaOmron  # noqa: E402
from scripts.eval_robocasa_gui import GuiWrapper  # noqa: E402


@pytest.mark.parametrize("scales", [{}, {"link2": 0.5}, {"link0": 1.2, "link4": 1.2, "link7": 0.8}])
def test_ghost_matches_nominal_full_model_and_preserves_live_state(scales):
    robot = PandaOmron(idn=0)
    robot.add_base(OmronMobileBase(idn=0))
    robot.add_gripper(PandaGripper(idn="0_right"))
    xml = robot.get_xml()
    live = mujoco.MjModel.from_xml_string(scale_robot_xml(xml, scales))
    data = mujoco.MjData(live)
    # Include mobile-base rotation/translation, torso travel, and finger positions.
    for name, value in {
        "mobilebase0_joint_mobile_forward": 1.2,
        "mobilebase0_joint_mobile_side": -0.7,
        "mobilebase0_joint_mobile_yaw": 0.8,
        "mobilebase0_joint_torso_height": 0.2,
    }.items():
        data.joint(name).qpos[:] = value
    data.qpos[-2:] = [0.02, -0.02]
    mujoco.mj_forward(live, data)
    robot_env = SimpleNamespace(
        robots=[SimpleNamespace(robot_model=robot)],
        sim=SimpleNamespace(model=SimpleNamespace(_model=live), data=SimpleNamespace(_data=data)),
    )
    ghost = NominalArmGhost(robot_env)
    scene = mujoco.MjvScene(live, maxgeom=500)
    original_state = data.qpos.copy(), data.qvel.copy(), data.ctrl.copy(), data.geom_xpos.copy()
    original_geom_count = live.ngeom
    nominal = mujoco.MjModel.from_xml_string(xml)
    nominal_data = mujoco.MjData(nominal)
    for offset in (0.0, 0.1):
        virtual_q = robot.init_qpos + offset
        nominal_data.qpos[:] = data.qpos
        for i, q in enumerate(virtual_q, 1):
            nominal_data.joint(f"robot0_joint{i}").qpos[:] = q
        mujoco.mj_forward(nominal, nominal_data)
        ghost.update(scene, virtual_q)
        assert scene.ngeom == len(ghost.geom_ids)
        for index, geom_id in enumerate(ghost.geom_ids):
            rendered = scene.geoms[index]
            expected_id = nominal.geom(ghost.model.geom(geom_id).name).id
            np.testing.assert_allclose(rendered.pos, nominal_data.geom_xpos[expected_id], atol=1e-6)
            np.testing.assert_allclose(
                rendered.mat.ravel(), nominal_data.geom_xmat[expected_id], atol=1e-6
            )
            assert rendered.rgba[3] == pytest.approx(0.3)
            assert rendered.objtype == mujoco.mjtObj.mjOBJ_UNKNOWN
            assert rendered.matid == -1
            if rendered.type == mujoco.mjtGeom.mjGEOM_MESH:
                mesh_name = nominal.mesh(nominal.geom_dataid[expected_id]).name
                assert live.mesh(rendered.dataid // 2).name == mesh_name
                assert "_x_emb_" not in mesh_name
    assert live.ngeom == original_geom_count
    for actual, expected in zip((data.qpos, data.qvel, data.ctrl, data.geom_xpos), original_state):
        np.testing.assert_array_equal(actual, expected)


def test_gui_updates_overlay_from_returned_observation_before_sync_and_rebuilds_on_reset(
    monkeypatch,
):
    import mujoco.viewer
    import scripts.eval_robocasa_gui as gui_module

    events = []
    viewers = []
    overlays = []

    class TestEnv(gym.Env):
        def __init__(self):
            self.env = SimpleNamespace(
                sim=SimpleNamespace(
                    model=SimpleNamespace(_model=None), data=SimpleNamespace(_data=None)
                ),
                render_collision_mesh=False,
                render_visual_mesh=True,
                renderer_config={"cam_config": {}},
                control_freq=20,
            )

        def reset(self, **kwargs):
            events.append("reset")
            return {"state.joint_position": np.full(7, 0.1)}, {}

        def step(self, action):
            events.append("step")
            return {"state.joint_position": np.full(7, 0.2)}, 0, False, False, {}

    class TestViewer:
        def __init__(self, *args):
            self.opt = SimpleNamespace(geomgroup=np.zeros(6))
            self.cam = SimpleNamespace()
            self.user_scn = object()
            self.locked = False
            self.closed = False
            viewers.append(self)

        @contextmanager
        def lock(self):
            self.locked = True
            yield
            self.locked = False

        def sync(self):
            events.append("sync")

        def is_running(self):
            return not self.closed

        def close(self):
            self.closed = True
            events.append("close")

    class TestOverlay:
        def __init__(self, env):
            self.qs = []
            overlays.append(self)

        def update(self, scene, q):
            assert viewers[-1].locked
            self.qs.append(q.copy())
            events.append("overlay")

    monkeypatch.setattr(mujoco.viewer, "launch_passive", TestViewer)
    monkeypatch.setattr(gui_module.time, "sleep", lambda _: None)
    env = GuiWrapper(TestEnv(), overlay_factory=TestOverlay)
    try:
        env.reset()
        env.step({})
        assert events == ["reset", "overlay", "sync", "step", "overlay", "sync"]
        np.testing.assert_array_equal(overlays[0].qs[0], np.full(7, 0.1))
        np.testing.assert_array_equal(overlays[0].qs[1], np.full(7, 0.2))
        env.reset()
        assert viewers[0].closed
        assert len(overlays) == 2
        assert events[-4:] == ["close", "reset", "overlay", "sync"]
    finally:
        env.close()
    assert viewers[-1].closed
