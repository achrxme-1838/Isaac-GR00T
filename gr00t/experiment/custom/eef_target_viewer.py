"""Draw policy targets in a shared native MuJoCo viewer overlay."""

import numpy as np


EEF_COLORS = {"left": (0.1, 0.9, 1.0, 1.0), "right": (1.0, 0.2, 0.75, 1.0)}
AXIS_COLORS = ((1.0, 0.15, 0.15, 1.0), (0.15, 1.0, 0.15, 1.0), (0.2, 0.4, 1.0, 1.0))


def eef_rotations_in_world(sim, base_body, wrist_bodies, action):
    """Decode first-two-ROWS rot6d with the policy's own conversion convention.

    The server already composes relative actions with the observation. Only the
    pelvis-to-world rotation remains. Degenerate 6D values have no orientation;
    return None so the viewer can report them without inventing identity axes.
    """
    from gr00t.data.state_action.pose import EndEffectorPose

    base_rot = sim.data.get_body_xmat(base_body)
    rotations = {}
    for side, body in wrist_bodies.items():
        command = action.get(f"action.{side}_wrist_eef_9d")
        target = None
        if command is not None:
            command = np.asarray(command, dtype=np.float64)
            if command.shape != (9,) or not np.all(np.isfinite(command)):
                raise ValueError(f"Invalid {side} EEF target: expected finite (9,), got {command}")
            rows = command[3:].reshape(2, 3)
            if np.linalg.norm(rows[0]) > 1e-8 and np.linalg.norm(np.cross(*rows)) > 1e-8:
                local = EndEffectorPose(rotation=command[3:], rotation_type="rot6d")
                target = base_rot @ local.rotation_matrix
        rotations[side] = (sim.data.get_body_xmat(body).copy(), target)
    return rotations


def rotation_distance_degrees(first, second):
    """Smallest SO(3) angle, avoiding Euler wraparound and gimbal ambiguity."""
    cosine = (np.trace(first.T @ second) - 1) / 2
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def eef_positions_in_world(sim, base_body, wrist_bodies, action):
    """Pair current wrists with absolute pelvis-frame targets decoded by the server.

    Missing EEF actions (e.g. a joint-only smoke test) have no target marker.
    The first three entries of XYZ_ROT6D are position in metres, not a delta.
    """
    base_pos = sim.data.get_body_xpos(base_body)
    base_rot = sim.data.get_body_xmat(base_body)
    positions = {}
    for side, body in wrist_bodies.items():
        target = action.get(f"action.{side}_wrist_eef_9d")
        if target is not None:
            target = np.asarray(target, dtype=np.float64)
            if target.shape != (9,) or not np.all(np.isfinite(target)):
                raise ValueError(f"Invalid {side} EEF target: expected finite (9,), got {target}")
            target = base_pos + base_rot @ target[:3]
        positions[side] = (sim.data.get_body_xpos(body).copy(), target)
    return positions


class EefTargetViewer:
    """Own the passive viewer's user scene without touching physics or camera observations."""

    def __init__(self, viewer, joint_ghost=None):
        # Native rendering remains optional for headless clients and CPU tests.
        import mujoco

        self.mujoco = mujoco
        self.viewer = viewer
        self.joint_ghost = joint_ghost

    def update(self, positions, joint_targets=None, rotations=None):
        if not self.viewer.is_running():
            return
        if self.joint_ghost is not None and joint_targets:
            self.joint_ghost.update(joint_targets)
        mj = self.mujoco
        with self.viewer.lock():
            scene = self.viewer.user_scn
            scene.ngeom = 0
            if self.joint_ghost is not None and joint_targets:
                self.joint_ghost.append_to(scene)

            def add_geom(kind, size, position, color, label=""):
                if scene.ngeom >= scene.maxgeom:
                    raise RuntimeError("MuJoCo viewer has no room for EEF markers")
                geom = scene.geoms[scene.ngeom]
                mj.mjv_initGeom(
                    geom, kind, np.asarray(size), position, np.eye(3).ravel(), np.asarray(color)
                )
                geom.category = mj.mjtCatBit.mjCAT_DECOR
                geom.transparent = 0
                geom.emission = 0.3
                geom.label = label
                scene.ngeom += 1
                return geom

            def add_axes(position, rotation, length, width):
                # Local XYZ axes in world space are the COLUMNS of R, even
                # though the model serializes its first two ROWS as rot6d.
                for axis, color in enumerate(AXIS_COLORS):
                    axis_geom = add_geom(mj.mjtGeom.mjGEOM_CAPSULE, [0.0] * 3, position, color)
                    end = position + length * rotation[:, axis]
                    mj.mjv_connector(axis_geom, mj.mjtGeom.mjGEOM_CAPSULE, width, position, end)

            for side, (current, target) in positions.items():
                label = "L" if side == "left" else "R"
                current_rot, target_rot = (rotations or {}).get(side, (None, None))
                if current_rot is not None:
                    add_axes(current, current_rot, 0.06, 0.004)
                if rotations is not None and self.joint_ghost is not None and joint_targets:
                    joint_pos, joint_rot = self.joint_ghost.wrist_pose(side)
                    add_axes(joint_pos, joint_rot, 0.09, 0.005)
                    joint_label = f"{label} joint FK"
                    if target_rot is not None:
                        difference = rotation_distance_degrees(joint_rot, target_rot)
                        joint_label += f" / EEF {difference:.1f} deg"
                    add_geom(
                        mj.mjtGeom.mjGEOM_SPHERE,
                        [0.008] * 3,
                        joint_pos,
                        EEF_COLORS[side],
                        joint_label,
                    )
                add_geom(
                    mj.mjtGeom.mjGEOM_SPHERE,
                    [0.01] * 3,
                    current,
                    [1.0, 1.0, 1.0, 1.0],
                    f"{label} wrist",
                )
                if target is None:
                    continue
                color = EEF_COLORS[side]
                distance = float(np.linalg.norm(target - current))
                target_label = f"{label} EEF target ({distance * 100:.1f} cm"
                if target_rot is not None:
                    add_axes(target, target_rot, 0.13, 0.006)
                    difference = rotation_distance_degrees(current_rot, target_rot)
                    target_label += f", {difference:.1f} deg"
                elif rotations is not None:
                    target_label += ", rot invalid"
                target_label += ")"
                add_geom(
                    mj.mjtGeom.mjGEOM_SPHERE,
                    [0.022] * 3,
                    target,
                    color,
                    target_label,
                )
                if distance > 1e-6:
                    line = add_geom(mj.mjtGeom.mjGEOM_LINE, [0.0] * 3, current, color)
                    mj.mjv_connector(line, mj.mjtGeom.mjGEOM_LINE, 3.0, current, target)
        # sync() locks internally; call it outside viewer.lock().
        self.viewer.sync()
