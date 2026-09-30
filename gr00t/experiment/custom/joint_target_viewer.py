"""Forward-kinematics preview of joint targets, isolated from simulator physics."""

from gr00t.experiment.custom.eef_target_viewer import EEF_COLORS
import numpy as np


class JointTargetGhost:
    """Render translucent arm/hand meshes at the applied joint targets.

    Waist targets are included in FK because they move the shoulders. The pelvis
    and all other uncommanded joints retain their actual positions. No dynamics
    are stepped and no values are written into the live model or data.
    """

    def __init__(self, sim, joint_names, qpos_indices):
        import mujoco

        self.mujoco = mujoco
        self.model = sim.model._model
        self.live_data = sim.data._data
        self.data = mujoco.MjData(self.model)
        self.qpos_indices = qpos_indices
        self.scene = mujoco.MjvScene(self.model, maxgeom=10000)
        self.option = mujoco.MjvOption()
        self.option.geomgroup[:] = 0
        self.option.geomgroup[1] = 1  # Visual meshes, not collision proxies.
        self.camera = mujoco.MjvCamera()
        self.geom_sides = {}
        self.wrist_body_ids = {}
        for side in ("left", "right"):
            wrist_joint = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_names[f"{side}_arm"][-1]
            )
            self.wrist_body_ids[side] = self.model.jnt_bodyid[wrist_joint]
            shoulder_joint = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_names[f"{side}_arm"][0]
            )
            root = self.model.jnt_bodyid[shoulder_joint]
            for geom_id, body_id in enumerate(self.model.geom_bodyid):
                if self.model.geom_group[geom_id] != 1:
                    continue
                while body_id and body_id != root:
                    body_id = self.model.body_parentid[body_id]
                if body_id == root:
                    self.geom_sides[geom_id] = side

    def update(self, targets):
        """Use the same absolute, clipped targets that reach the controllers."""
        self.data.qpos[:] = self.live_data.qpos
        self.data.mocap_pos[:] = self.live_data.mocap_pos
        self.data.mocap_quat[:] = self.live_data.mocap_quat
        for group, values in targets.items():
            self.data.qpos[self.qpos_indices[group]] = values
        self.mujoco.mj_kinematics(self.model, self.data)
        self.mujoco.mj_camlight(self.model, self.data)
        self.mujoco.mjv_updateScene(
            self.model,
            self.data,
            self.option,
            None,
            self.camera,
            self.mujoco.mjtCatBit.mjCAT_ALL,
            self.scene,
        )

    def append_to(self, scene):
        """Append arm meshes to the shared viewer scene while its lock is held."""
        mj = self.mujoco
        for source in self.scene.geoms[: self.scene.ngeom]:
            if source.objtype != mj.mjtObj.mjOBJ_GEOM:
                continue
            side = self.geom_sides.get(source.objid)
            if side is None:
                continue
            if scene.ngeom >= scene.maxgeom:
                raise RuntimeError("MuJoCo viewer has no room for joint target arms")
            dest = scene.geoms[scene.ngeom]
            color = np.array([*EEF_COLORS[side][:3], 0.3], dtype=np.float32)
            mj.mjv_initGeom(dest, source.type, source.size, source.pos, source.mat.ravel(), color)
            # Use the existing mesh resource, but override its opaque material.
            dest.dataid = source.dataid
            dest.modelrbound = source.modelrbound
            dest.matid = -1
            dest.category = mj.mjtCatBit.mjCAT_DECOR
            dest.transparent = 1
            dest.emission = 0.2
            scene.ngeom += 1

    def wrist_pose(self, side):
        """World pose of the wrist yaw frame at the last applied joint targets."""
        body = self.wrist_body_ids[side]
        return self.data.xpos[body].copy(), self.data.xmat[body].reshape(3, 3).copy()
