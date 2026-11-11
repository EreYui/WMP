# SPDX-License-Identifier: Apache-2.0
"""合并非脚部固定链接，显式守恒质量、质心和惯量。"""

import copy
import math
import os
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np


def _pose(element):
    if element is None:
        return np.eye(4)
    x, y, z = map(float, element.get("rpy", "0 0 0").split())
    cx, sx, cy, sy, cz, sz = math.cos(x), math.sin(x), math.cos(y), math.sin(y), math.cos(z), math.sin(z)
    matrix = np.eye(4)
    matrix[:3, :3] = (
        (cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx),
        (sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx),
        (-sy, cy * sx, cy * cx),
    )
    matrix[:3, 3] = list(map(float, element.get("xyz", "0 0 0").split()))
    return matrix


def _origin(element, transform):
    old = element.find("origin")
    if old is not None:
        element.remove(old)
    r = transform[:3, :3]
    angles = (
        math.atan2(r[2, 1], r[2, 2]),
        math.atan2(-r[2, 0], math.hypot(r[0, 0], r[1, 0])),
        math.atan2(r[1, 0], r[0, 0]),
    )
    ET.SubElement(element, "origin", xyz=" ".join(map(str, transform[:3, 3])), rpy=" ".join(map(str, angles)))


def _inertia(link, transform=np.eye(4)):
    node = link.find("inertial")
    if node is None:
        return 0.0, np.zeros(3), np.zeros((3, 3))
    mass = float(node.find("mass").get("value"))
    pose = transform @ _pose(node.find("origin"))
    values = node.find("inertia").attrib
    tensor = np.array([[float(values[f"i{a}{b}" if a <= b else f"i{b}{a}"]) for b in "xyz"] for a in "xyz"])
    return mass, pose[:3, 3], pose[:3, :3] @ tensor @ pose[:3, :3].T


def prepare_reference_urdf(source, destination):
    source, destination = Path(source), Path(destination)
    root = ET.parse(source).getroot()
    # 固定根附件和空肩链接合并；保留命名脚刚体以维持接触观测。
    for joint in list(root.findall("joint")):
        if joint.get("type") != "fixed" or joint.get("dont_collapse") == "true":
            continue
        links = {link.get("name"): link for link in root.findall("link")}
        parent_name, child_name = joint.find("parent").get("link"), joint.find("child").get("link")
        parent, child = links[parent_name], links[child_name]
        transform = _pose(joint.find("origin"))
        first, second = _inertia(parent), _inertia(child, transform)
        mass = first[0] + second[0]
        if mass:
            center = (first[0] * first[1] + second[0] * second[1]) / mass
            tensor = np.zeros((3, 3))
            for m, c, inertia in (first, second):
                delta = c - center
                tensor += inertia + m * (np.dot(delta, delta) * np.eye(3) - np.outer(delta, delta))
            old = parent.find("inertial")
            if old is not None:
                parent.remove(old)
            node = ET.SubElement(parent, "inertial")
            ET.SubElement(node, "origin", xyz=" ".join(map(str, center)), rpy="0 0 0")
            ET.SubElement(node, "mass", value=str(mass))
            ET.SubElement(
                node,
                "inertia",
                **{f"i{a}{b}": str(tensor[i, j]) for i, a in enumerate("xyz") for j, b in enumerate("xyz") if i <= j},
            )
        for tag in ("visual", "collision"):
            for node in child.findall(tag):
                node = copy.deepcopy(node)
                _origin(node, transform @ _pose(node.find("origin")))
                parent.append(node)
        for descendant in root.findall("joint"):
            if descendant.find("parent").get("link") == child_name:
                descendant.find("parent").set("link", parent_name)
                _origin(descendant, transform @ _pose(descendant.find("origin")))
        root.remove(joint)
        root.remove(child)
    destination.parent.mkdir(parents=True, exist_ok=True)
    for mesh in root.findall(".//mesh"):
        original = (source.parent / mesh.get("filename")).resolve()
        mesh.set("filename", os.path.relpath(original, destination.parent))
    ET.indent(root)
    ET.ElementTree(root).write(destination, encoding="utf-8", xml_declaration=True)
    return destination


def configure_reference_colliders(path):
    """显式解除碰撞网格实例化，并记录参考接触偏置。"""
    from pxr import PhysxSchema, Usd, UsdPhysics

    stage = Usd.Stage.Open(str(path))
    for _ in range(4):
        instances = [prim for prim in stage.Traverse() if prim.IsInstance()]
        if not instances:
            break
        for prim in instances:
            prim.SetInstanceable(False)
    count = 0
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            collision = PhysxSchema.PhysxCollisionAPI.Apply(prim)
            collision.CreateContactOffsetAttr(0.01)
            collision.CreateRestOffsetAttr(0.0)
            count += 1
    if not count:
        raise RuntimeError(f"No authored collision shapes in {path}")
    stage.GetRootLayer().Save()
    return count
