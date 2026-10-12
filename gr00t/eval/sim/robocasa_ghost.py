"""Nominal Panda geometry for a passive viewer, separate from simulation/cameras."""

from copy import deepcopy
import xml.etree.ElementTree as ET

from gr00t.eval.sim.robocasa_exp_adaptation import ARM_JOINTS
import mujoco
import numpy as np


class NominalArmGhost:
    """Draw the original arm at the exact joint angles supplied to the policy."""

    def __init__(self, robot_env):
        self.live_model = robot_env.sim.model._model
        self.live_data = robot_env.sim.data._data
        xml = ET.fromstring(robot_env.robots[0].robot_model.get_xml())
        arm = deepcopy(xml.find(".//body[@name='robot0_link0']"))
        arm.set("pos", "0 0 0")
        arm.set("quat", "1 0 0 0")
        # Keep the original meshes and fingers, but isolate the arm from the base.
        for child in list(xml):
            if child.tag not in ("compiler", "default", "asset", "worldbody"):
                xml.remove(child)
        worldbody = xml.find("worldbody")
        worldbody.clear()
        worldbody.append(arm)
        self.model = mujoco.MjModel.from_xml_string(ET.tostring(xml, encoding="unicode"))
        self.data = mujoco.MjData(self.model)
        self.root_id = self.live_model.body("robot0_link0").id
        self.arm_qpos = [self.model.joint(name).qposadr[0] for name in ARM_JOINTS]
        # Copy finger positions from the actual robot, before replacing arm angles.
        self.live_qpos = [
            self.live_model.joint(self.model.joint(i).name).qposadr[0]
            for i in range(self.model.njnt)
        ]
        self.ghost_qpos = self.model.jnt_qposadr.copy()
        self.geom_ids = np.flatnonzero(self.model.geom_group == 1)
        self.mesh_dataids = {}
        for geom_id in self.geom_ids:
            if self.model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_MESH:
                name = self.model.mesh(self.model.geom_dataid[geom_id]).name
                # x_emb retains the original assets alongside the stretched copies.
                # mjvGeom uses 2*mesh_id for the mesh, 2*mesh_id+1 for its hull.
                self.mesh_dataids[geom_id] = 2 * self.live_model.mesh(name).id

    def update(self, scene, joint_position, *, start=0):
        """Draw geoms (or append at start); caller must hold the viewer lock."""
        self.data.qpos[self.ghost_qpos] = self.live_data.qpos[self.live_qpos]
        self.data.qpos[self.arm_qpos] = joint_position
        mujoco.mj_kinematics(self.model, self.data)
        root_pos = self.live_data.xpos[self.root_id]
        root_rot = self.live_data.xmat[self.root_id].reshape(3, 3)
        if start + len(self.geom_ids) > scene.maxgeom:
            raise ValueError("Viewer scene has insufficient capacity for the nominal arm.")
        scene.ngeom = start
        for geom_id in self.geom_ids:
            geom = scene.geoms[scene.ngeom]
            mujoco.mjv_initGeom(
                geom,
                type=int(self.model.geom_type[geom_id]),
                size=self.model.geom_size[geom_id],
                pos=root_pos + root_rot @ self.data.geom_xpos[geom_id],
                mat=(root_rot @ self.data.geom_xmat[geom_id].reshape(3, 3)).ravel(),
                rgba=np.array([0.1, 0.8, 1.0, 0.3], dtype=np.float32),
            )
            geom.dataid = self.mesh_dataids.get(geom_id, -1)
            geom.category = mujoco.mjtCatBit.mjCAT_DECOR
            geom.objtype = mujoco.mjtObj.mjOBJ_UNKNOWN
            geom.objid = -1
            geom.segid = -1
            geom.matid = -1
            geom.emission = 0.15
            geom.specular = 0.2
            geom.modelrbound = self.model.geom_rbound[geom_id]
            geom.transparent = True
            scene.ngeom += 1
