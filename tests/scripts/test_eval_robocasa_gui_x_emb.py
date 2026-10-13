"""CLI validation and compiled Panda geometry checks (no viewer or policy server)."""

import argparse

from gr00t.eval.sim.robocasa_x_emb import parse_x_emb, scale_robot_xml
import numpy as np
import pytest


@pytest.mark.parametrize(
    "value",
    [
        "[]",
        '{"link2": 0}',
        '{"link4": -1}',
        '{"link2": true}',
        '{"link2": "2"}',
        '{"link2": NaN}',
        '{"link2": Infinity}',
        '{"link1": 2}',
        '{"link5": 2}',
        "not json",
    ],
)
def test_reject_invalid_link_scales(value):
    with pytest.raises(argparse.ArgumentTypeError):
        parse_x_emb(value)


def test_unit_scales_preserve_xml_exactly():
    xml = '<mujoco model="unchanged" />'
    assert scale_robot_xml(xml, parse_x_emb("{}")) == xml
    assert scale_robot_xml(xml, parse_x_emb('{"link2": 1, "link4": 1.0}')) == xml


@pytest.mark.parametrize(
    "flag", ["--use_exp_adap", "--use-exp-adap", "--use_ext_adap", "--use-ext-adap"]
)
def test_expert_adaptation_cli_aliases(flag):
    from scripts.eval_robocasa_gui_x_emb import parse_args

    assert parse_args([flag]).use_exp_adap
    assert parse_args([f"{flag}=True"]).use_exp_adap
    assert not parse_args([f"{flag}=False"]).use_exp_adap
    assert not parse_args([]).use_exp_adap


@pytest.mark.parametrize("flag", ["--sync_init", "--sync-init"])
def test_sync_init_cli_aliases(flag):
    from scripts.eval_robocasa_gui_x_emb import parse_args

    assert not parse_args([]).sync_init
    assert parse_args([flag]).sync_init
    assert parse_args([f"{flag}=True"]).sync_init
    assert not parse_args([f"{flag}=False"]).sync_init


@pytest.fixture(scope="module")
def panda_xml():
    pytest.importorskip("robosuite")
    from robosuite.models.robots.compositional import PandaOmron

    return PandaOmron(idn=0).get_xml()


def _geom_vertices(model, geom_id, mujoco):
    mesh_id = model.geom_dataid[geom_id]
    start = model.mesh_vertadr[mesh_id]
    vertices = model.mesh_vert[start : start + model.mesh_vertnum[mesh_id]]
    rotation = np.empty(9)
    mujoco.mju_quat2Mat(rotation, model.geom_quat[geom_id])
    return vertices @ rotation.reshape(3, 3).T + model.geom_pos[geom_id]


@pytest.mark.parametrize(
    "link, child, axis",
    [
        ("link0", "link1", 2),
        ("link2", "link3", 1),
        ("link3", "link4", 0),
        ("link4", "link5", 1),
        ("link6", "link7", 0),
        ("link7", "right_hand", 2),
    ],
)
@pytest.mark.parametrize("ratio", [0.8, 2.0])
def test_compiled_link_lengths_meshes_and_dynamics(panda_xml, link, child, axis, ratio):
    mujoco = pytest.importorskip("mujoco")
    original = mujoco.MjModel.from_xml_string(panda_xml)
    modified = mujoco.MjModel.from_xml_string(scale_robot_xml(panda_xml, {link: ratio}))
    stretch = np.ones(3)
    stretch[axis] = ratio
    body_id = original.body(f"robot0_{link}").id
    child_id = original.body(f"robot0_{child}").id

    expected_positions = original.body_pos.copy()
    expected_positions[child_id] *= stretch
    np.testing.assert_allclose(modified.body_pos, expected_positions)
    np.testing.assert_allclose(modified.body_ipos[body_id], original.body_ipos[body_id] * stretch)
    np.testing.assert_array_equal(modified.body_mass, original.body_mass)
    assert np.all(modified.body_inertia[body_id] > 0)

    # Compare the actual vertices in body coordinates after MuJoCo mesh compilation.
    geom_ids = np.flatnonzero(original.geom_bodyid == body_id)
    assert set(original.geom_group[geom_ids]) == {0, 1}
    for geom_id in geom_ids:
        np.testing.assert_allclose(
            _geom_vertices(modified, geom_id, mujoco),
            _geom_vertices(original, geom_id, mujoco) * stretch,
            atol=1e-6,
        )

    for field in ("jnt_pos", "jnt_axis", "jnt_range", "actuator_ctrlrange", "qpos0"):
        np.testing.assert_array_equal(getattr(modified, field), getattr(original, field))
    assert (modified.nq, modified.nv, modified.nu) == (original.nq, original.nv, original.nu)

    data = mujoco.MjData(modified)
    for _ in range(20):
        mujoco.mj_step(modified, data)
    assert np.isfinite(data.qpos).all()
    assert np.isfinite(data.qvel).all()
