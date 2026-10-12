"""Map Panda joint observations through changed-arm FK and nominal-arm pose IK."""

from copy import deepcopy
import warnings
import xml.etree.ElementTree as ET

from gr00t.eval._ik_metrics import IKDiagnostics
from gr00t.eval.sim.robocasa_x_emb import scale_robot_xml
import gymnasium as gym
import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


ARM_JOINTS = tuple(f"robot0_joint{i}" for i in range(1, 8))
POSITION_TOLERANCE = 1e-4  # metres
ROTATION_TOLERANCE = 1e-3  # radians
ROTATION_WEIGHT = 0.2


def _skew(vector):
    x, y, z = vector
    return np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])


def _rotation_log_jacobian(rotvec):
    """Derivative of Log(exp(delta) R) with respect to a spatial rotation delta."""
    angle = np.linalg.norm(rotvec)
    cross = _skew(rotvec)
    coefficient = (
        1 / 12 + angle**2 / 720
        if angle < 1e-4
        else (1 - 0.5 * angle / np.tan(0.5 * angle)) / angle**2
    )
    return np.eye(3) - 0.5 * cross + coefficient * cross @ cross


class PandaKinematics:
    """Seven-joint arm in robot0_link0 coordinates, including the real grip-site offset."""

    def __init__(self, robot_xml: str, grip_site: str):
        arm = deepcopy(ET.fromstring(robot_xml).find(".//body[@name='robot0_link0']"))
        arm.set("pos", "0 0 0")
        arm.set("quat", "1 0 0 0")
        # Isolate kinematics: no meshes, rendering, actuators, base or finger DOFs.
        for body in arm.iter("body"):
            for child in list(body):
                keep = (
                    child.tag in ("body", "inertial")
                    or (child.tag == "joint" and child.get("name") in ARM_JOINTS)
                    or (child.tag == "site" and child.get("name") == grip_site)
                )
                if not keep:
                    body.remove(child)
        root = ET.Element("mujoco")
        ET.SubElement(root, "compiler", angle="radian")
        ET.SubElement(root, "worldbody").append(arm)
        self.model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
        self.data = mujoco.MjData(self.model)
        self.site_id = self.model.site(grip_site).id
        joint_ids = [self.model.joint(name).id for name in ARM_JOINTS]
        self.qpos_indices = self.model.jnt_qposadr[joint_ids]
        self.dof_indices = self.model.jnt_dofadr[joint_ids]
        self.lower, self.upper = self.model.jnt_range[joint_ids].T.copy()

    def pose(self, q):
        self.data.qpos[self.qpos_indices] = q
        mujoco.mj_kinematics(self.model, self.data)
        return (
            self.data.site_xpos[self.site_id].copy(),
            self.data.site_xmat[self.site_id].reshape(3, 3).copy(),
        )

    def jacobian(self, q):
        self.pose(q)
        mujoco.mj_comPos(self.model, self.data)
        jac_pos = np.zeros((3, self.model.nv))
        jac_rot = np.zeros_like(jac_pos)
        mujoco.mj_jacSite(self.model, self.data, jac_pos, jac_rot, self.site_id)
        return np.vstack((jac_pos, jac_rot))[:, self.dof_indices]

    def pose_error(self, q, target_pos, target_rot):
        position, rotation = self.pose(q)
        return np.concatenate(
            (position - target_pos, Rotation.from_matrix(rotation @ target_rot.T).as_rotvec())
        )

    def inverse(self, target_pos, target_rot, seed):
        """Bounded pose IK, softly preferring the previous solution for redundancy."""
        seed = np.clip(seed, self.lower, self.upper)
        weights = np.array([1, 1, 1, ROTATION_WEIGHT, ROTATION_WEIGHT, ROTATION_WEIGHT])
        posture_weight = 1e-4

        def residual(q):
            pose_error = self.pose_error(q, target_pos, target_rot)
            return np.concatenate((weights * pose_error, posture_weight * (q - seed)))

        def jacobian(q):
            error = self.pose_error(q, target_pos, target_rot)
            jac = self.jacobian(q)
            jac[3:] = _rotation_log_jacobian(error[3:]) @ jac[3:]
            return np.vstack((weights[:, None] * jac, posture_weight * np.eye(7)))

        initial_error = self.pose_error(seed, target_pos, target_rot)
        if np.linalg.norm(initial_error) < 1e-10:
            return seed, initial_error
        result = least_squares(
            residual,
            seed,
            jac=jacobian,
            bounds=(self.lower, self.upper),
            max_nfev=80,
            ftol=1e-9,
            xtol=1e-9,
            gtol=1e-9,
        )
        return result.x, self.pose_error(result.x, target_pos, target_rot)


class ExpertAdaptationWrapper(gym.Wrapper):
    """Adapt observations before MultiStepWrapper builds the policy's temporal history.

    Both kinematic models use the same arm-root frame, so mobile-base and torso
    motion cancel. Grip-site position AND orientation are matched. RoboCasa's
    EEF quaternion uses the hand body, whose fixed rotation to the grip site is
    identical in both embodiments. Neither the live simulator nor actions change.
    """

    def __init__(self, env, scales):
        super().__init__(env)
        self.scales = dict(scales)
        self.enabled = any(ratio != 1 for ratio in scales.values())
        self.nominal = None
        self.modified = None
        self.previous_q = None
        self.warned_ik = False
        self.ik_diagnostics = IKDiagnostics()

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.previous_q = None
        self.warned_ik = False
        self.ik_diagnostics = IKDiagnostics()
        if self.enabled:
            robot_env = self.unwrapped.env
            robot = robot_env.robots[0]
            # The source robot XML stays nominal; x_emb only changes compiled sim XML.
            xml = robot.robot_model.get_xml()
            grip_site = robot_env.sim.model.site_id2name(robot.eef_site_id["right"])
            self.nominal = PandaKinematics(xml, grip_site)
            self.modified = PandaKinematics(scale_robot_xml(xml, self.scales), grip_site)
        return self._adapt(obs, info)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs, info = self._adapt(obs, info)
        return obs, reward, terminated, truncated, info

    def _adapt(self, obs, info):
        if not self.enabled:
            return obs, info
        q_real = np.asarray(obs["state.joint_position"], dtype=np.float64)
        target_pos, target_rot = self.modified.pose(q_real)
        seed = q_real if self.previous_q is None else self.previous_q
        q_nominal, error = self.nominal.inverse(target_pos, target_rot, seed)
        self.previous_q = q_nominal.copy()
        position_error = float(np.linalg.norm(error[:3]))
        rotation_error = float(np.linalg.norm(error[3:]))
        converged = position_error <= POSITION_TOLERANCE and rotation_error <= ROTATION_TOLERANCE
        if not converged and not self.warned_ik:
            warnings.warn(
                f"Expert adaptation IK outside tolerance: {position_error:.4f} m, "
                f"{rotation_error:.4f} rad. Using a joint-limited approximation; "
                "the pose may be outside the nominal arm's workspace.",
                stacklevel=2,
            )
            self.warned_ik = True

        # Match the arm's task-space twist with a damped velocity inverse.
        qvel = np.asarray(obs["state.joint_velocity"], dtype=np.float64)
        jac_real = self.modified.jacobian(q_real)
        jac_nominal = self.nominal.jacobian(q_nominal)
        weights = np.array([1, 1, 1, ROTATION_WEIGHT, ROTATION_WEIGHT, ROTATION_WEIGHT])
        jac = weights[:, None] * jac_nominal
        twist_error = weights * ((jac_real - jac_nominal) @ qvel)
        qvel_nominal = qvel + jac.T @ np.linalg.solve(jac @ jac.T + 1e-6 * np.eye(6), twist_error)

        adapted = dict(obs)
        for key, value in {
            "state.joint_position": q_nominal,
            "state.joint_position_cos": np.cos(q_nominal),
            "state.joint_position_sin": np.sin(q_nominal),
            "state.joint_velocity": qvel_nominal,
        }.items():
            adapted[key] = value.astype(obs[key].dtype)
        info = dict(info)
        info.update(
            exp_adap_ik_converged=converged,
            exp_adap_position_error_m=position_error,
            exp_adap_rotation_error_rad=rotation_error,
        )
        info.update(self.ik_diagnostics.record(position_error, rotation_error, converged))
        return adapted, info
