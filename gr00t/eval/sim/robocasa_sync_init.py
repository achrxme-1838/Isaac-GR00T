"""Match each changed Panda's initial grip-site pose to a nominal RoboCasa reset."""

from copy import deepcopy

from gr00t.eval.sim.robocasa_exp_adaptation import (
    ARM_JOINTS,
    POSITION_TOLERANCE,
    ROTATION_TOLERANCE,
    PandaKinematics,
)
from gr00t.eval.sim.robocasa_x_emb import scale_robot_xml
import gymnasium as gym
import numpy as np


class SyncInitialPoseWrapper(gym.Wrapper):
    """Reset nominally, then rebuild the same scene with changed links and solve arm IK.

    RoboCasa settles objects and the robot during reset. Capturing that actual
    nominal state avoids using already-drifted changed-arm joints as the reference.
    Only arm qpos/qvel are replaced; scene, base, torso and fingers retain their
    nominal reset state. This wrapper must precede recording/adaptation wrappers.
    It owns link scaling, so do not also call configure_x_emb on this environment.
    """

    def __init__(self, env, scales):
        super().__init__(env)
        self.scales = dict(scales)
        self.enabled = any(ratio != 1 for ratio in self.scales.values())
        self._apply_scales = False
        if self.enabled:
            robot_env = self.unwrapped.env
            robot_env.hard_reset = True
            robot_env.set_xml_processor(self._process_xml)

    def _process_xml(self, xml):
        return scale_robot_xml(xml, self.scales) if self._apply_scales else xml

    def reset(self, **kwargs):
        self._apply_scales = False
        obs, info = self.env.reset(**kwargs)
        if not self.enabled:
            return obs, info

        robot_env = self.unwrapped.env
        robot = robot_env.robots[0]
        sim = robot_env.sim
        model = sim.model._model
        joint_ids = [model.joint(name).id for name in ARM_JOINTS]
        qpos_indices = model.jnt_qposadr[joint_ids].copy()
        dof_indices = model.jnt_dofadr[joint_ids].copy()
        site_id = robot.eef_site_id["right"]
        target_world = sim.data.site_xpos[site_id].copy()
        root_pos = sim.data.get_body_xpos("robot0_link0")
        root_rotation = sim.data.get_body_xmat("robot0_link0").reshape(3, 3)
        target_pos = root_rotation.T @ (target_world - root_pos)
        target_rot = root_rotation.T @ sim.data.site_xmat[site_id].reshape(3, 3)
        changed = PandaKinematics(
            scale_robot_xml(robot.robot_model.get_xml(), self.scales), model.site(site_id).name
        )
        q, error = changed.inverse(target_pos, target_rot, sim.data.qpos[qpos_indices])
        position_error = float(np.linalg.norm(error[:3]))
        rotation_error = float(np.linalg.norm(error[3:]))
        if (
            not np.isfinite(error).all()
            or position_error > POSITION_TOLERANCE
            or rotation_error > ROTATION_TOLERANCE
        ):
            raise RuntimeError(
                f"--sync_init could not match the nominal initial gripper pose: "
                f"{position_error:.6f} m, {rotation_error:.6f} rad. "
                "Try less extreme --x_emb ratios or disable --sync_init."
            )

        state = sim.get_state()
        ctrl = sim.data.ctrl.copy()
        xml = robot_env.model.get_xml()  # Unprocessed XML: apply length ratios exactly once.
        rng_state = deepcopy(robot_env.rng.bit_generator.state)
        numpy_state = np.random.get_state()
        deterministic_reset = robot_env.deterministic_reset
        self._apply_scales = True
        try:
            # Rebuild controllers/references for the new model without resampling a scene.
            robot_env.reset_from_xml_string(xml)
        finally:
            robot_env.rng.bit_generator.state = rng_state
            np.random.set_state(numpy_state)
            robot_env.deterministic_reset = deterministic_reset

        sim = robot_env.sim
        sim.set_state(state)
        sim.data.ctrl[:] = ctrl
        sim.data.qpos[qpos_indices] = q
        sim.data.qvel[dof_indices] = 0
        sim.data._data.qacc_warmstart[:] = 0
        sim.forward()
        robot = robot_env.robots[0]
        robot.composite_controller.update_state()
        for controller in robot.part_controllers.values():
            controller.update(force=True)
        robot.part_controllers["right"].update_initial_joints(q)
        robot.composite_controller.reset()

        robot_env._obs_cache.clear()
        source = robot_env.viewer if robot_env.viewer_get_obs else robot_env
        raw_obs = source._get_observations(force_update=True)
        obs = self.unwrapped.get_groot_observation(self.unwrapped.get_basic_observation(raw_obs))
        info = {
            **info,
            "sync_init_position_error_m": position_error,
            "sync_init_rotation_error_rad": rotation_error,
        }
        print(
            f"  Synced initial gripper: {target_world.round(5)} m (world), "
            f"IK error {position_error:.6f} m / {rotation_error:.6f} rad",
            flush=True,
        )
        return obs, info
