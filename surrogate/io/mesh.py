"""Read a tetrahedral mesh, and pull the graph structure out of it.

The ``.vtk`` file holds two things: where every node is, and which groups of
four nodes form a tetrahedron. Everything the network needs about the shape is
derived from that second list.

Three derivations live here:

``surface_triangles``
    the outer skin - triangles belonging to only one tetrahedron

``surface_edges``
    which pairs of surface nodes are joined by material

``vertex_normals``
    which way the skin faces at each node, and how sharply it turns

Run on its own to check one bracket::

    python -m simjeb.mesh 0
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

# meshio is NOT imported here, and that is deliberate.
#
# One function in this module needs it: read(), which parses a .vtk file. That
# happens in stage 1 and nowhere else. Stages 3 to 5 - training, scoring,
# plotting - import this module only because graph.py does, and they never
# open a mesh: the features were built once and cached.
#
# Imported at the top, a machine that only ever trains would still need a mesh
# format library installed, and would fail at import with a
# ModuleNotFoundError naming a library it has no use for. That is exactly what
# happened on Kaggle.
#
# So the import sits inside read(), where the dependency is real.


class MeshError(Exception):
    """The mesh cannot be used. The caller should exclude this bracket."""


@dataclass
class Mesh:
    """One tetrahedral mesh: node positions and element connectivity."""

    points: np.ndarray        # (n_nodes, 3) coordinates in mm
    tetrahedra: np.ndarray    # (n_tets, 4) node indices, 0-based

    @property
    def n_nodes(self):
        return len(self.points)

    @property
    def n_tetrahedra(self):
        return len(self.tetrahedra)


def read(path):
    """Load a ``.vtk``. Raises :class:`MeshError` if it holds no tetrahedra."""
    import meshio       # see the note beside the imports

    path = Path(path)
    raw = meshio.read(path)

    if "tetra" not in raw.cells_dict:
        raise MeshError(f"{path.name} contains no tetrahedra")

    points = np.asarray(raw.points, dtype=np.float64)
    tetrahedra = raw.cells_dict["tetra"].astype(np.int64)
    return Mesh(points=points, tetrahedra=tetrahedra)


# ---------------------------------------------------------------------------
# THE OUTER SKIN
# ---------------------------------------------------------------------------

# The four triangular faces of a tetrahedron, written so that each one's corner
# order points its normal outward. The order matters: reverse a triangle and
# its normal points into the material instead of out of it.
TETRAHEDRON_FACES = [
    [0, 2, 1],
    [0, 1, 3],
    [1, 2, 3],
    [0, 3, 2],
]


def surface_triangles(mesh):
    """The triangles on the outside of the solid.

    A face inside the material is shared by two tetrahedra. A face on the
    outside belongs to only one. So counting how often each triangle appears
    separates skin from interior, with no geometry involved at all.
    """
    faces = []
    for corners in TETRAHEDRON_FACES:
        faces.append(mesh.tetrahedra[:, corners])
    faces = np.concatenate(faces)

    # Sorting each triangle's three corner numbers makes the same triangle
    # look identical however it happened to be written down. The sorted copy is
    # only for counting - the returned triangles keep their original order, so
    # their normals still point outward.
    sorted_faces = np.sort(faces, axis=1)
    _, first_seen, counts = np.unique(
        sorted_faces, axis=0, return_index=True, return_counts=True)

    on_the_outside = counts == 1
    return faces[first_seen[on_the_outside]]


# ---------------------------------------------------------------------------
# THE EDGES
# ---------------------------------------------------------------------------

# The six edges of a tetrahedron: every pair of its four corners.
TETRAHEDRON_EDGES = [
    [0, 1], [0, 2], [0, 3],
    [1, 2], [1, 3], [2, 3],
]


def surface_edges(mesh, keep):
    """Undirected edges between kept nodes, taken from the tetrahedra.

    ``keep`` is a True/False array, one entry per mesh node.

    Why tetrahedron edges and not the skin triangles' edges
    -------------------------------------------------------
    Every triangle edge is also a tetrahedron edge, so this is a superset. The
    extra ones cut straight THROUGH the material - from one face of a thin rib
    to the node opposite it. Material genuinely joins those two nodes and load
    genuinely passes between them.

    What it refuses to do matters just as much. Two surfaces facing each other
    across a gap share no tetrahedron, so no edge appears. A rule like "connect
    anything within 2 mm" would bridge that gap and teach the network a
    connection that does not exist.
    """
    pairs = []
    for corners in TETRAHEDRON_EDGES:
        pairs.append(mesh.tetrahedra[:, corners])
    pairs = np.concatenate(pairs)

    # An edge is only usable if both of its ends survive.
    first_end_kept = keep[pairs[:, 0]]
    second_end_kept = keep[pairs[:, 1]]
    pairs = pairs[first_end_kept & second_end_kept]

    if len(pairs) == 0:
        raise MeshError("no edges between the kept nodes")

    # The same edge appears in several tetrahedra, and as (a,b) in one and
    # (b,a) in another. Sorting each row makes those identical so np.unique can
    # remove the duplicates.
    pairs = np.sort(pairs, axis=1)
    return np.unique(pairs, axis=0)


def node_degrees(edges, n_nodes):
    """How many edges touch each node.

    Every edge is counted at both of its ends, which is what makes the count
    equal the number of neighbours. ``minlength`` guarantees one slot per node,
    so a node with no edges at all shows up as a zero instead of being left off
    the end of the array.
    """
    both_ends = edges.ravel()
    return np.bincount(both_ends, minlength=n_nodes)


# ---------------------------------------------------------------------------
# NORMALS AND SHARPNESS
# ---------------------------------------------------------------------------

def vertex_normals(points, triangles):
    """Which way the surface faces at each node, and how sharply it turns.

    Returns ``(normals, sharpness)``.

    ``normals`` is a unit-length arrow per node, pointing out of the material.
    It tells the network whether a node sits on a flat face, the side of a rib
    or inside a hole.

    ``sharpness`` runs from 1.0 on a flat surface down towards 0 at a sharp
    edge, and it comes free from the same arithmetic - see below.
    """
    corner_a = points[triangles[:, 0]]
    corner_b = points[triangles[:, 1]]
    corner_c = points[triangles[:, 2]]

    # The cross product of two of a triangle's edges is an arrow at right
    # angles to both: straight out of that triangle. Its length is twice the
    # triangle's area, so summing these un-normalised arrows weights each
    # triangle by its size for free.
    face_normals = np.cross(corner_b - corner_a, corner_c - corner_a)

    # A node sits on several triangles, so add up the arrows of all of them.
    # Each triangle contributes to its three corners, so its normal is repeated
    # three times to line up with triangles.ravel().
    n_nodes = len(points)
    flat_corners = triangles.ravel()
    normals = np.zeros((n_nodes, 3))
    for axis in range(3):
        repeated = np.repeat(face_normals[:, axis], 3)
        normals[:, axis] = np.bincount(flat_corners, weights=repeated,
                                       minlength=n_nodes)

    # Sharpness, before the arrows are rescaled.
    #
    # If a node's triangles all face the same way, their arrows add up fully
    # and the sum is long. If they face different ways they partly cancel and
    # the sum is short. So
    #
    #     length of the sum  /  sum of the lengths
    #
    # is 1.0 on a flat surface and falls towards 0 at a sharp edge. That is
    # curvature, and curvature is what sets stress concentration. Rescaling the
    # normals to unit length destroys it, so it is captured first.
    summed_length = np.linalg.norm(normals, axis=1, keepdims=True)

    face_areas = np.repeat(np.linalg.norm(face_normals, axis=1), 3)
    total_length = np.bincount(flat_corners, weights=face_areas,
                               minlength=n_nodes)
    total_length = total_length.reshape(-1, 1)

    sharpness = summed_length / np.maximum(total_length, 1e-12)

    # Now scale to unit length: the direction is what matters, not the size of
    # the triangles that produced it.
    normals = normals / np.maximum(summed_length, 1e-12)

    return normals, sharpness



