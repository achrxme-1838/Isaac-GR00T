"""GUI recording options and the streaming encoder, without a display or policy server."""

from types import SimpleNamespace

import numpy as np
import pytest


mujoco = pytest.importorskip("mujoco")

from gr00t.eval.sim.gui_video_recorder import (  # noqa: E402
    GuiVideoRecorder,
    RoboCasaCameraRenderer,
    recording_directory,
)
from scripts.eval_robocasa_gui import parse_args as gui_args  # noqa: E402
from scripts.eval_robocasa_gui_x_emb import parse_args as x_emb_args  # noqa: E402


@pytest.mark.parametrize("parse_args", [gui_args, x_emb_args])
@pytest.mark.parametrize("flag", ["--record_vid", "--record-vid"])
def test_recording_opt_in_and_custom_output(parse_args, flag, tmp_path):
    assert recording_directory(parse_args([])) is None
    assert recording_directory(parse_args(["--video-dir", str(tmp_path)])) is None
    args = parse_args([flag, "--video-dir", str(tmp_path)])
    assert recording_directory(args) == tmp_path
    assert args.record_camera == "robot0_agentview_left"
    assert recording_directory(parse_args([flag])) != recording_directory(parse_args([flag]))


@pytest.mark.parametrize("with_overlay", [False, True])
def test_recording_runs_all_episodes_without_viewer_or_realtime_sleep(
    monkeypatch, tmp_path, with_overlay
):
    import gymnasium as gym
    import mujoco.viewer
    import scripts.eval_robocasa_gui as gui_module

    recorders = []

    class TestEnv(gym.Env):
        def __init__(self):
            self.env = SimpleNamespace()

        def observation(self):
            return {
                "state.joint_position": np.full(7, self.steps),
                "annotation.human.action.task_description": "open drawer",
            }

        def reset(self, **kwargs):
            self.steps = 0
            return self.observation(), {}

        def step(self, action):
            self.steps += 1
            return (
                self.observation(),
                0,
                self.steps == 2,
                False,
                {
                    "n_env_steps": 1,
                    "success": True,
                },
            )

    class TestRecorder:
        def __init__(self, env, path, *, camera, overlay):
            self.closed = False
            self.frames = []
            self.overlay = overlay
            recorders.append(self)

        def write(self, obs):
            self.frames.append(obs["state.joint_position"].copy())

        def close(self):
            self.closed = True

    def unexpected(*args, **kwargs):
        pytest.fail("Recording must not open a viewer or wait for real-time playback")

    monkeypatch.setattr(mujoco.viewer, "launch_passive", unexpected)
    monkeypatch.setattr(gui_module.time, "sleep", unexpected)
    monkeypatch.setattr(gui_module, "GuiVideoRecorder", TestRecorder)
    gui = gui_module.GuiWrapper(
        TestEnv(),
        video_dir=tmp_path,
        overlay_factory=(lambda env: object()) if with_overlay else None,
    )
    policy = SimpleNamespace(reset=lambda: None, get_action=lambda obs: ({}, {}))
    try:
        gui_module.run_episodes(gui, gui, policy, n_episodes=2, seed=0)
        assert gui.viewer is None
        assert len(recorders) == 2
        for recorder in recorders:
            assert recorder.closed
            assert (recorder.overlay is not None) == with_overlay
            np.testing.assert_array_equal(recorder.frames, [np.full(7, n) for n in range(3)])
    finally:
        gui.close()


def test_renderer_reuses_context_and_keeps_policy_scene_separate(monkeypatch):
    model = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><camera name="record" pos="0 0 2"/>'
        '<geom type="sphere" size="0.1"/></worldbody></mujoco>'
    )
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    events = []
    policy_scene = object()
    context = SimpleNamespace(
        gl_ctx=SimpleNamespace(make_current=lambda: events.append("make_current")),
        con=SimpleNamespace(offWidth=64, offHeight=64),
        scn=policy_scene,
    )

    def no_new_context(*args, **kwargs):
        pytest.fail("Recording must not create another OpenGL context alongside policy cameras")

    def resize(width, height, con):
        assert con is context.con
        con.offWidth, con.offHeight = width, height
        events.append("resize")

    def render(viewport, scene, con):
        assert con is context.con
        assert scene is not policy_scene
        events.append("render")

    def read(rgb, depth, viewport, con):
        rgb[:32] = [255, 0, 0]
        rgb[32:] = [0, 0, 255]
        events.append("read")

    monkeypatch.setattr(mujoco, "Renderer", no_new_context)
    monkeypatch.setattr(mujoco, "MjrContext", no_new_context)
    monkeypatch.setattr(mujoco, "mjr_resizeOffscreen", resize)
    monkeypatch.setattr(mujoco, "mjr_setBuffer", lambda *args: None)
    monkeypatch.setattr(mujoco, "mjr_render", render)
    monkeypatch.setattr(mujoco, "mjr_readPixels", read)
    renderer = RoboCasaCameraRenderer(model, context, height=64, width=128)
    try:
        renderer.update_scene(data, camera="record", scene_option=mujoco.MjvOption())
        assert renderer.scene.ngeom > 0
        image = renderer.render()
        np.testing.assert_array_equal(image[0, 0], [0, 0, 255])
        np.testing.assert_array_equal(image[-1, 0], [255, 0, 0])
        assert context.scn is policy_scene
        assert events == ["make_current", "resize", "render", "read"]
    finally:
        renderer.close()
    # The borrowed context remains owned and usable by the environment.
    context.gl_ctx.make_current()
    assert context.con.offWidth == 128
    assert context.scn is policy_scene


@pytest.mark.parametrize("with_overlay", [False, True])
def test_mp4_finalized_with_correct_rgb_fps_and_overlay(tmp_path, monkeypatch, with_overlay):
    import shutil

    import gr00t.eval.sim.gui_video_recorder as recording

    cv2 = pytest.importorskip("cv2")
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg is required for encoding")

    class FakeRenderer:
        def __init__(self, model, context, *, height, width):
            self.frame = np.empty((height, width, 3), dtype=np.uint8)
            self.scene = SimpleNamespace(ngeom=1)
            self.closed = False

        def update_scene(self, data, *, camera, scene_option):
            assert camera == "robot0_agentview_left"
            assert scene_option.geomgroup[0] == 0
            self.scene.ngeom = 1
            self.frame[:] = [255, 0, 0]

        def render(self):
            if self.scene.ngeom == 2:
                self.frame[:, self.frame.shape[1] // 2 :] = [0, 0, 255]
            return self.frame

        def close(self):
            self.closed = True

    class Overlay:
        def update(self, scene, q, *, start):
            assert start == 1  # Append, preserving the real robot's scene.
            np.testing.assert_array_equal(q, np.arange(7))
            scene.ngeom += 1

    monkeypatch.setattr(recording, "RoboCasaCameraRenderer", FakeRenderer)
    model = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><camera name="robot0_agentview_left" /></worldbody></mujoco>'
    )
    data = mujoco.MjData(model)
    robot = SimpleNamespace(
        sim=SimpleNamespace(
            model=SimpleNamespace(_model=model),
            data=SimpleNamespace(_data=data),
            _render_context_offscreen=object(),
        ),
        render_collision_mesh=False,
        render_visual_mesh=True,
        control_freq=20,
    )
    path = tmp_path / "episode_001.mp4"
    recorder = GuiVideoRecorder(robot, path, overlay=Overlay() if with_overlay else None)
    renderer = recorder.renderer
    process = recorder.process
    try:
        for _ in range(4):
            recorder.write({"state.joint_position": np.arange(7)})
    finally:
        recorder.close()
    recorder.close()  # Episode end and GUI teardown may both finalize.
    assert renderer.closed
    assert process.poll() == 0
    video = cv2.VideoCapture(str(path))
    try:
        assert video.get(cv2.CAP_PROP_FPS) == pytest.approx(20)
        assert video.get(cv2.CAP_PROP_FRAME_COUNT) == 4
        ok, bgr = video.read()
        assert ok and bgr.shape == (720, 960, 3)
        rgb = bgr[:, :, ::-1]
        np.testing.assert_allclose(rgb[100, 100], [255, 0, 0], atol=5)
        np.testing.assert_allclose(
            rgb[100, 800], [0, 0, 255] if with_overlay else [255, 0, 0], atol=5
        )
    finally:
        video.release()
