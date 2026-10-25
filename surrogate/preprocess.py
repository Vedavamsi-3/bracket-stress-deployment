"""Raw .vtk + .fem  ->  model inputs, exactly as in training, without field.csv.

Same logic as the training repo's graph.build() + features.build(), with one
change: surface nodes come from the tetrahedra (nodes on outer faces), not
from the 'surf' column of field.csv. Verified identical on bracket 0.
"""
from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from surrogate.io import deck as deck_module
from surrogate.io import mesh as mesh_module

NODE_ORDINARY = 0
NODE_CLAMPED = 1
NODE_LOADED = 2


class PreprocessError(Exception):
    """The files cannot be turned into a valid graph."""


@dataclass
class Graph:
    positions: np.ndarray      # (n, 3) mm, centred on the interfaces
    edges: np.ndarray          # (n_edges, 2) each edge once
    node_kind: np.ndarray      # (n,) ordinary / clamped / loaded
    triangles: np.ndarray      # (n_tri, 3) the skin, graph numbering
    landmarks: np.ndarray      # (5, 3) bolt-hole and lug centres, centred
    mesh_node_ids: np.ndarray  # (n,) original 1-based node id of each graph node


@dataclass
class Features:
    node_features: np.ndarray  # (n, 9) float32
    edge_index: np.ndarray     # (2, 2*n_edges) int64, sender / receiver
    edge_features: np.ndarray  # (2*n_edges, 4) float32


def build_graph(vtk_path, fem_path):
    try:
        deck = deck_module.parse(fem_path, model_id=-1)
        mesh = mesh_module.read(vtk_path)
    except (deck_module.DeckError, mesh_module.MeshError) as error:
        raise PreprocessError(str(error)) from error

    if deck.n_mesh_nodes != mesh.n_nodes:
        raise PreprocessError(
            f".fem describes {deck.n_mesh_nodes:,} nodes but .vtk has {mesh.n_nodes:,}")

    # THE CHANGE: surface = every node on an outer face of the tetrahedra
    skin = mesh_module.surface_triangles(mesh)
    keep = np.zeros(mesh.n_nodes, dtype=bool)
    keep[np.unique(skin)] = True
    kept_rows = np.flatnonzero(keep)

    edges_mesh = mesh_module.surface_edges(mesh, keep)
    mesh_to_graph = np.full(mesh.n_nodes, -1, dtype=np.int64)
    mesh_to_graph[kept_rows] = np.arange(len(kept_rows))
    edges = mesh_to_graph[edges_mesh]
    triangles = mesh_to_graph[skin]

    degrees = mesh_module.node_degrees(edges, len(kept_rows))
    if (degrees == 0).any():
        raise PreprocessError(f"{int((degrees == 0).sum())} surface nodes have no edges")

    landmarks = interface_landmarks(mesh.points, deck)
    origin = landmarks.mean(axis=0)

    node_kind = np.full(len(kept_rows), NODE_ORDINARY, dtype=np.int64)
    clamped = mesh_to_graph[deck.fixed_rows]
    loaded = mesh_to_graph[deck.loaded_rows]
    if (clamped < 0).any() or (loaded < 0).any():
        raise PreprocessError("a clamped or loaded node is not on the surface")
    node_kind[clamped] = NODE_CLAMPED
    node_kind[loaded] = NODE_LOADED

    return Graph(
        positions=(mesh.points - origin)[kept_rows],
        edges=edges,
        node_kind=node_kind,
        triangles=triangles,
        landmarks=landmarks - origin,
        mesh_node_ids=kept_rows + 1,
    )


def interface_landmarks(points, deck):
    """Centres of the 4 bolt holes, then the centre of the load lug."""
    centres = [points[deck.bolt_hole_rows(i)].mean(axis=0)
               for i in range(len(deck.bolt_holes))]
    centres.append(points[deck.loaded_rows].mean(axis=0))
    return np.vstack(centres)


def build_features(graph):
    normals, sharpness = mesh_module.vertex_normals(graph.positions, graph.triangles)
    kind_one_hot = np.eye(3)[graph.node_kind]
    to_clamp = distance_to_nearest(graph.positions, graph.node_kind == NODE_CLAMPED)
    to_load = distance_to_nearest(graph.positions, graph.node_kind == NODE_LOADED)
    node_features = np.column_stack([kind_one_hot, normals, sharpness, to_clamp, to_load])

    senders = np.concatenate([graph.edges[:, 0], graph.edges[:, 1]])
    receivers = np.concatenate([graph.edges[:, 1], graph.edges[:, 0]])
    step = graph.positions[receivers] - graph.positions[senders]
    length = np.linalg.norm(step, axis=1, keepdims=True)

    return Features(
        node_features=node_features.astype(np.float32),
        edge_index=np.vstack([senders, receivers]).astype(np.int64),
        edge_features=np.hstack([step, length]).astype(np.float32),
    )


def distance_to_nearest(positions, is_target):
    targets = positions[is_target]
    if len(targets) == 0:
        raise PreprocessError("no nodes of that kind to measure the distance to")
    distances, _ = cKDTree(targets).query(positions)
    return distances.reshape(-1, 1)


def preprocess(vtk_path, fem_path):
    """The one function the worker calls: raw files -> (graph, features)."""
    graph = build_graph(vtk_path, fem_path)
    return graph, build_features(graph)