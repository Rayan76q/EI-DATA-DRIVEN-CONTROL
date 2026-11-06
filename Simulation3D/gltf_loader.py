"""Lecture d'un fichier glTF (.gltf + .bin + textures) pour l'afficher dans meshcat, qui ne sait pas le faire seul."""
import json
from pathlib import Path

import numpy as np
import meshcat.geometry as g
import meshcat.transformations as tf

COMPONENT_TYPES = {5121: np.uint8, 5123: np.uint16, 5125: np.uint32, 5126: np.float32}
TYPE_SIZES = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}
REPEAT_WRAPPING = 1000  # constante three.js


class TexturedMeshGeometry(g.Geometry):
    """Comme g.TriangularMeshGeometry, avec en plus les normales et les coordonnées de texture (UV)."""

    def __init__(self, vertices, normals, uvs, faces):
        super().__init__()
        self.attributes = {"position": vertices, "normal": normals, "uv": uvs}
        self.faces = faces

    def lower(self, object_data):
        return {
            "uuid": self.uuid,
            "type": "BufferGeometry",
            "data": {
                "attributes": {name: g.pack_numpy_array(values.T) for name, values in self.attributes.items()},
                "index": g.pack_numpy_array(self.faces.T),
            },
        }


class TexturedMaterial(g.MeshPhongMaterial):
    """Matériau Phong qui accepte aussi une normal map et une texture émissive."""

    def __init__(self, normalMap=None, emissiveMap=None, **kwargs):
        super().__init__(**kwargs)
        self.extra_maps = {"normalMap": normalMap, "emissiveMap": emissiveMap}

    def lower(self, object_data):
        data = super().lower(object_data)
        for field, texture in self.extra_maps.items():
            if texture is not None:
                data[field] = texture.lower_in_object(object_data)
        return data


def load_gltf_parts(path):
    """Renvoie une liste de couples (géométrie, matériau) à passer à set_object, un par partie visible du modèle."""
    path = Path(path)
    gltf = json.loads(path.read_text(encoding="utf-8"))
    buffers = [(path.parent / buffer["uri"]).read_bytes() for buffer in gltf["buffers"]]

    parts = []
    for node_index, transform in _mesh_nodes(gltf):
        for primitive in gltf["meshes"][gltf["nodes"][node_index]["mesh"]]["primitives"]:
            material = gltf["materials"][primitive["material"]]
            if _is_invisible(material):
                continue
            attributes = primitive["attributes"]
            geometry = TexturedMeshGeometry(
                vertices=_transform_points(_read_accessor(gltf, buffers, attributes["POSITION"]), transform),
                normals=_transform_normals(_read_accessor(gltf, buffers, attributes["NORMAL"]), transform),
                uvs=_read_accessor(gltf, buffers, attributes["TEXCOORD_0"]),
                faces=_read_accessor(gltf, buffers, primitive["indices"]).astype(np.uint32).reshape(-1, 3),
            )
            parts.append((geometry, _build_material(gltf, material, path.parent)))
    return parts


def _mesh_nodes(gltf):
    """Parcourt l'arbre de la scène : renvoie (indice du noeud, transformation globale) pour chaque noeud avec un maillage."""
    found = []

    def visit(index, parent_transform):
        node = gltf["nodes"][index]
        transform = parent_transform @ _node_transform(node)
        if "mesh" in node:
            found.append((index, transform))
        for child in node.get("children", []):
            visit(child, transform)

    for root in gltf["scenes"][gltf.get("scene", 0)]["nodes"]:
        visit(root, np.eye(4))
    return found


def _node_transform(node):
    if "matrix" in node:
        return np.array(node["matrix"], dtype=float).reshape(4, 4).T  # glTF stocke les matrices par colonnes
    x, y, z, w = node.get("rotation", [0, 0, 0, 1])
    scale = np.diag([*node.get("scale", [1, 1, 1]), 1.0])
    return tf.translation_matrix(node.get("translation", [0, 0, 0])) @ tf.quaternion_matrix([w, x, y, z]) @ scale


def _read_accessor(gltf, buffers, accessor_index):
    accessor = gltf["accessors"][accessor_index]
    view = gltf["bufferViews"][accessor["bufferView"]]
    dtype = np.dtype(COMPONENT_TYPES[accessor["componentType"]])
    size = TYPE_SIZES[accessor["type"]]
    return np.ndarray(
        shape=(accessor["count"], size),
        dtype=dtype,
        buffer=buffers[view["buffer"]],
        offset=view.get("byteOffset", 0) + accessor.get("byteOffset", 0),
        strides=(view.get("byteStride", size * dtype.itemsize), dtype.itemsize),
    ).copy()


def _transform_points(points, transform):
    return (points @ transform[:3, :3].T + transform[:3, 3]).astype(np.float32)


def _transform_normals(normals, transform):
    normals = normals @ np.linalg.inv(transform[:3, :3])  # équivaut à appliquer l'inverse de la transposée
    return (normals / np.linalg.norm(normals, axis=1, keepdims=True)).astype(np.float32)


def _is_invisible(material):
    base_color = material["pbrMetallicRoughness"].get("baseColorFactor", [1, 1, 1, 1])
    return material.get("alphaMode") == "BLEND" and base_color[3] == 0


def _build_material(gltf, material, folder):
    pbr = material["pbrMetallicRoughness"]
    base_color = pbr.get("baseColorFactor", [1, 1, 1, 1])
    blended = material.get("alphaMode") == "BLEND"

    def texture(info):
        return None if info is None else _load_texture(gltf, info["index"], folder)

    return TexturedMaterial(
        color=_to_hex(base_color[:3]),
        map=texture(pbr.get("baseColorTexture")),
        normalMap=texture(material.get("normalTexture")),
        emissiveMap=texture(material.get("emissiveTexture")),
        emissive=_to_hex(material.get("emissiveFactor", [0, 0, 0])),
        normalScale=[1, -1],  # convention des normal maps glTF dans three.js, sans tangentes
        transparent=blended,
        opacity=base_color[3] if blended else 1.0,
        depthWrite=not blended,
    )


def _load_texture(gltf, texture_index, folder):
    image = gltf["images"][gltf["textures"][texture_index]["source"]]
    # PngImage convient aussi aux JPEG : le navigateur détecte le vrai format de l'image.
    return g.ImageTexture(
        g.PngImage.from_file(folder / image["uri"]),
        wrap=[REPEAT_WRAPPING, REPEAT_WRAPPING],
        flipY=False,  # l'origine des UV glTF est en haut à gauche de l'image
    )


def _to_hex(rgb):
    r, g_, b = (int(round(channel * 255)) for channel in rgb)
    return (r << 16) | (g_ << 8) | b
