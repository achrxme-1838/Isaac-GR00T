"""PandaOmron link-length changes shared by GUI and recorded rollouts."""

import argparse
from copy import deepcopy
from functools import partial
import json
import math
import xml.etree.ElementTree as ET

import numpy as np


LINK_AXES = {"link0": 2, "link2": 1, "link3": 0, "link4": 1, "link6": 0, "link7": 2}


def parse_x_emb(value: str) -> dict[str, float]:
    try:
        scales = json.loads(value)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(f"Invalid JSON: {exc}") from exc
    if not isinstance(scales, dict):
        raise argparse.ArgumentTypeError("--x_emb must be a JSON object, e.g. '{\"link2\": 1.2}'.")
    for link, ratio in scales.items():
        if link not in LINK_AXES:
            raise argparse.ArgumentTypeError(
                f"Unsupported link {link!r}; choose from: {', '.join(LINK_AXES)}."
            )
        if (
            isinstance(ratio, bool)
            or not isinstance(ratio, (int, float))
            or not math.isfinite(ratio)
            or ratio <= 0
        ):
            raise argparse.ArgumentTypeError(f"{link}: ratio must be a finite positive number.")
    return scales


def _vec(value):
    return np.fromstring(value, sep=" ")


def _text(value):
    return " ".join(f"{component:.12g}" for component in value)


def scale_robot_xml(xml: str, scales: dict[str, float]) -> str:
    """Stretch stock Panda links in the XML before MuJoCo builds physics/controllers."""
    scales = {link: ratio for link, ratio in scales.items() if ratio != 1}
    if not scales:
        return xml
    root = ET.fromstring(xml)
    assets = root.find("asset")
    meshes = {mesh.get("name"): mesh for mesh in assets.findall("mesh")}
    for link, ratio in scales.items():
        body = root.find(f".//body[@name='robot0_{link}']")
        if body is None:
            raise ValueError(
                f"robot0_{link} is missing; --x_emb requires the stock PandaOmron robot."
            )
        stretch = np.ones(3)
        stretch[LINK_AXES[link]] = ratio

        # Only direct attachments move; downstream links keep their own lengths.
        for child in body:
            if "pos" in child.attrib:
                child.set("pos", _text(_vec(child.get("pos")) * stretch))

        scaled_meshes = {}
        for geom in body.findall("geom"):
            mesh_name = geom.get("mesh")
            if mesh_name not in scaled_meshes:
                mesh = deepcopy(meshes[mesh_name])
                new_name = f"{mesh_name}_x_emb_{link}"
                mesh.set("name", new_name)
                mesh.set("scale", _text(_vec(mesh.get("scale", "1 1 1")) * stretch))
                assets.append(mesh)
                scaled_meshes[mesh_name] = new_name
            geom.set("mesh", scaled_meshes[mesh_name])

        # Stock Panda inertias are diagonal in the body frame. Affine stretch at fixed mass:
        # C = integral(r r^T dm), I = trace(C) * identity - C.
        inertial = body.find("inertial")
        inertia = _vec(inertial.get("diaginertia"))
        second_moments = (inertia.sum() / 2 - inertia) * stretch**2
        inertial.set("diaginertia", _text(second_moments.sum() - second_moments))
    return ET.tostring(root, encoding="unicode")


def configure_x_emb(env, scales):
    """Apply length changes to each fresh model on reset, before controllers initialize."""
    robot_env = env.unwrapped.env
    robot_env.hard_reset = True
    robot_env.set_xml_processor(partial(scale_robot_xml, scales=scales))
