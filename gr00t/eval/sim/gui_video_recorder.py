"""Record a dedicated MuJoCo camera, including optional viewer-only geometry."""

from datetime import datetime
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np


DEFAULT_RECORD_CAMERA = "robot0_agentview_left"


def add_recording_args(parser):
    parser.add_argument(
        "--record_vid",
        "--record-vid",
        action="store_true",
        help="Disable the GUI and save one MP4 per episode.",
    )
    parser.add_argument("--video-dir", "--video_dir", type=Path, help="Directory for episode MP4s.")
    parser.add_argument(
        "--record-camera",
        "--record_camera",
        default=DEFAULT_RECORD_CAMERA,
        help="MuJoCo camera used for recording (default: robot0_agentview_left).",
    )


def recording_directory(args):
    if not args.record_vid:
        return None
    return args.video_dir or Path("eval_videos") / f"gui_{datetime.now():%y%m%d-%H%M%S-%f}"


class RoboCasaCameraRenderer:
    """Borrow RoboSuite's EGL context, with a separate scene for recording overlays."""

    def __init__(self, model, context, *, height, width):
        import mujoco

        if context is None:
            raise RuntimeError("GUI recording requires enable_render=True on the RoboCasa env.")
        self.model = model
        self.context = context
        self.scene = mujoco.MjvScene(model, maxgeom=10000)
        self.camera = mujoco.MjvCamera()
        self.camera.type = mujoco.mjtCamera.mjCAMERA_FIXED
        self.viewport = mujoco.MjrRect(0, 0, width, height)
        self.pixels = np.empty((height, width, 3), dtype=np.uint8)

    def update_scene(self, data, *, camera, scene_option):
        import mujoco

        self.camera.fixedcamid = self.model.camera(camera).id
        mujoco.mjv_updateScene(
            self.model,
            data,
            scene_option,
            None,
            self.camera,
            mujoco.mjtCatBit.mjCAT_ALL,
            self.scene,
        )

    def render(self):
        import mujoco

        context = self.context
        context.gl_ctx.make_current()
        width, height = self.viewport.width, self.viewport.height
        if width > context.con.offWidth or height > context.con.offHeight:
            # Resize only the framebuffer; keep the environment's textures and GL context.
            mujoco.mjr_resizeOffscreen(
                max(width, context.con.offWidth), max(height, context.con.offHeight), context.con
            )
        mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, context.con)
        mujoco.mjr_render(self.viewport, self.scene, context.con)
        mujoco.mjr_readPixels(self.pixels, None, self.viewport, context.con)
        return self.pixels[::-1]  # OpenGL reads bottom-up; the encoder expects top-down RGB.

    def close(self):
        # The environment owns the GL resources and still needs them for policy images.
        self.context = None
        self.scene = None
        self.pixels = None


class GuiVideoRecorder:
    """Stream RGB frames at the control frequency; never change policy observations."""

    def __init__(self, robot_env, path, *, camera=DEFAULT_RECORD_CAMERA, overlay=None):
        import mujoco

        self.path = Path(path)
        self.camera = camera
        self.overlay = overlay
        self.data = robot_env.sim.data._data
        self.renderer = None
        self.process = None
        model = robot_env.sim.model._model
        model.camera(camera)  # Validate the name before allocating rendering resources.
        system_ffmpeg = Path("/usr/bin/ffmpeg")
        ffmpeg = str(system_ffmpeg) if system_ffmpeg.is_file() else shutil.which("ffmpeg")
        if ffmpeg is None:
            raise RuntimeError("--record_vid requires ffmpeg. Install it or omit --record_vid.")
        encoder_env = os.environ.copy()
        if ffmpeg == str(system_ffmpeg):
            # Avoid the Conda/system C++ library mismatch, only in the encoder process.
            encoder_env.pop("LD_LIBRARY_PATH", None)
            encoder_env.pop("LD_PRELOAD", None)
        width, height = 960, 720
        self.option = mujoco.MjvOption()
        self.option.geomgroup[0] = int(robot_env.render_collision_mesh)
        self.option.geomgroup[1] = int(robot_env.render_visual_mesh)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.renderer = RoboCasaCameraRenderer(
            model, robot_env.sim._render_context_offscreen, height=height, width=width
        )
        try:
            self.process = subprocess.Popen(
                [
                    ffmpeg,
                    "-n",
                    "-loglevel",
                    "error",
                    "-f",
                    "rawvideo",
                    "-pix_fmt",
                    "rgb24",
                    "-s",
                    f"{width}x{height}",
                    "-r",
                    str(robot_env.control_freq),
                    "-i",
                    "pipe:0",
                    "-an",
                    "-c:v",
                    "libx264",
                    "-preset",
                    "fast",
                    "-crf",
                    "18",
                    "-pix_fmt",
                    "yuv420p",
                    "-threads",
                    "2",
                    "-movflags",
                    "+faststart",
                    str(self.path),
                ],
                stdin=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=encoder_env,
            )
        except Exception:
            self.renderer.close()
            self.renderer = None
            raise
        print(f"  Recording ({camera}): {self.path.resolve()}", flush=True)

    def write(self, obs):
        self.renderer.update_scene(self.data, camera=self.camera, scene_option=self.option)
        if self.overlay is not None:
            self.overlay.update(
                self.renderer.scene,
                obs["state.joint_position"],
                start=self.renderer.scene.ngeom,
            )
        frame = self.renderer.render()
        try:
            self.process.stdin.write(np.ascontiguousarray(frame).tobytes())
        except BrokenPipeError:
            self.close()  # Report the encoder error instead of just "Broken pipe".
            raise

    def close(self):
        process, self.process = self.process, None
        try:
            if process is not None:
                try:
                    _, stderr = process.communicate(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.communicate()
                    raise RuntimeError("Video encoder did not finish within 30 seconds.")
                if process.returncode:
                    raise RuntimeError(
                        f"Video recording failed: {stderr.decode('utf-8', errors='replace').strip()}"
                    )
                print(f"  Video saved: {self.path.resolve()}", flush=True)
        finally:
            if self.renderer is not None:
                self.renderer.close()
                self.renderer = None
