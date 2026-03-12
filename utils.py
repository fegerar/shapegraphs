"""
Utilities for computing shape graphs from 2D point sets.
"""

from typing import Dict, List, Tuple
from collections import defaultdict
import math
import pickle
import numpy as np
import networkx as nx
from scipy.spatial import Delaunay  # pylint: disable=no-name-in-module

def _angle_at_vertex(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    """Angle ∠apb at vertex p formed by rays pa and pb."""
    va = a - p
    vb = b - p
    cos_val = np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-15)
    cos_val = np.clip(cos_val, -1.0, 1.0)
    return math.acos(cos_val)


def compute_delaunay_edges(points: np.ndarray) -> List[Tuple[int, int]]:
    """Return the list of unique edges from a Delaunay triangulation."""
    if len(points) < 3:
        return []
    tri = Delaunay(points)
    edges = set()
    for simplex in tri.simplices:
        for i in range(3):
            for j in range(i + 1, 3):
                e = (min(simplex[i], simplex[j]), max(simplex[i], simplex[j]))
                edges.add(e)
    return list(edges)


def _build_face_structure(edges: List[Tuple[int, int]],
                          points: np.ndarray):
    """
    Build the face structure of the planar Delaunay graph.

    Returns:
      - faces: list of lists of vertex indices (counterclockwise).
               The last face is the outer (unbounded) face.
      - edge_to_faces: dict mapping (u,v) with u<v to list of face indices
    """
    # Build adjacency with angular ordering
    adj: Dict[int, List[int]] = defaultdict(list)
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    # Sort adjacency lists by angle
    for u in adj:
        adj[u].sort(key=lambda v: math.atan2(
            points[v][1] - points[u][1],
            points[v][0] - points[u][0]))

    # Next-edge function for face traversal (using half-edges)
    # For directed edge u->v, the next edge in the face to the LEFT
    # is v -> prev_neighbor_of_v_before_u
    def next_half_edge(u: int, v: int) -> Tuple[int, int]:
        neighbors = adj[v]
        idx = neighbors.index(u)
        # Previous neighbor in counterclockwise order around v
        prev_idx = (idx - 1) % len(neighbors)
        return v, neighbors[prev_idx]

    visited_half_edges = set()
    faces = []
    edge_to_faces: Dict[Tuple[int, int], List[int]] = defaultdict(list)

    for u, v in edges:
        for start_u, start_v in [(u, v), (v, u)]:
            if (start_u, start_v) in visited_half_edges:
                continue
            # Trace face
            face = []
            cu, cv = start_u, start_v
            while True:
                if (cu, cv) in visited_half_edges:
                    break
                visited_half_edges.add((cu, cv))
                face.append(cu)
                cu, cv = next_half_edge(cu, cv)
            if len(face) >= 2:
                face_idx = len(faces)
                faces.append(face)
                # Register edges
                for i, v in enumerate(face):
                    v_1 = face[(i + 1) % len(face)]
                    key = (min(v, v_1), max(v, v_1))
                    edge_to_faces[key].append(face_idx)

    return faces, edge_to_faces


def _angular_stability_in_face(p_idx: int, q_idx: int,
                               face: List[int],
                               points: np.ndarray) -> float:
    """
    Compute the angular stability contribution from one face.

    For a triangular face, this is the angle ∠prq where r is the third vertex.
    For a larger face, it is min over all other vertices r_i of ∠pr_iq.
    For the outer face (when the edge is on the convex hull), return 0.
    """
    other_verts = [v for v in face if v != p_idx and v != q_idx]
    if len(other_verts) == 0:
        return 0.0  # outer face / degenerate
    p = points[p_idx]
    q = points[q_idx]
    min_angle = math.pi
    for r_idx in other_verts:
        r = points[r_idx]
        angle = _angle_at_vertex(r, p, q)
        min_angle = min(min_angle, angle)
    return min_angle


def _compute_edge_stability(edge: Tuple[int, int],
                            edge_to_faces: dict,
                            faces: list,
                            points: np.ndarray) -> float:
    """
    Compute the angular stability of a Delaunay edge.

    stability = π − (α_left + α_right)
    where α_left, α_right are the max angles from each side.

    For the shape graph generalization (non-triangular faces):
    α(p,q) on one side = min angle ∠prq over all vertices r in that face.
    """
    p_idx, q_idx = edge
    face_indices = edge_to_faces.get(edge, [])

    alpha_sum = 0.0
    for fi in face_indices:
        alpha_sum += _angular_stability_in_face(p_idx, q_idx, faces[fi], points)

    # If edge is on convex hull (only one internal face), one side contributes 0
    stability = math.pi - alpha_sum
    return stability


def compute_shape_graph(points: np.ndarray,
                        alpha_threshold: float = math.pi / 4
                        ) -> List[Tuple[int, int]]:
    """
    Compute the shape graph of a 2D point set.

    Algorithm 1 from Brandes et al. (2025):
      1. Compute Delaunay triangulation
      2. Iteratively remove the least stable edge if its stability < α_threshold
      3. After each removal, update the face structure and recompute stabilities
         for edges in the merged face
      4. Stop when all remaining edges have stability ≥ α_threshold

    Parameters
    ----------
    points : (n, 2) array of 2D coordinates
    alpha_threshold : minimum angular stability (default π/4 = 45°)

    Returns
    -------
    List of edges (i, j) with i < j that form the shape graph.
    """
    if len(points) < 3:
        # Trivial cases
        if len(points) == 2:
            return [(0, 1)]
        return []

    edges = compute_delaunay_edges(points)
    if not edges:
        return []

    # Build face structure
    faces, edge_to_faces = _build_face_structure(edges, points)

    # Compute initial stabilities
    stabilities: Dict[Tuple[int, int], float] = {}
    for e in edges:
        stabilities[e] = _compute_edge_stability(e, edge_to_faces, faces, points)

    active_edges = set(edges)

    while True:
        # Find the least stable edge among active edges
        min_stab = math.pi
        min_edge = None
        for e in active_edges:
            s = stabilities.get(e, math.pi)
            if s < min_stab:
                min_stab = s
                min_edge = e

        if min_edge is None or min_stab >= alpha_threshold:
            break

        # Remove this edge and merge its two incident faces
        active_edges.discard(min_edge)

        face_indices = edge_to_faces.get(min_edge, [])
        if len(face_indices) == 2:
            f1, f2 = face_indices

            # Merge faces: combine vertices, remove duplicates, maintain order
            merged_verts = list(dict.fromkeys(faces[f1] + faces[f2]))
            # Sort by angle from centroid for consistent ordering
            cx = np.mean([points[v][0] for v in merged_verts])
            cy = np.mean([points[v][1] for v in merged_verts])
            merged_verts.sort(
                key=lambda v: math.atan2(points[v][1] - cy, points[v][0] - cx))

            new_face_idx = len(faces)
            faces.append(merged_verts)

            # Update edge_to_faces for all edges in the merged face
            for i, v in enumerate(merged_verts):
                v_1 = merged_verts[(i + 1) % len(merged_verts)]
                key = (min(v, v_1), max(v, v_1))
                if key in edge_to_faces and key in active_edges:
                    # Replace old face references with new merged face
                    new_face_list = []
                    for fi in edge_to_faces[key]:
                        if fi == f1 or fi == f2:
                            if new_face_idx not in new_face_list:
                                new_face_list.append(new_face_idx)
                        else:
                            new_face_list.append(fi)
                    edge_to_faces[key] = new_face_list

                    # Recompute stability for this edge
                    stabilities[key] = _compute_edge_stability(
                        key, edge_to_faces, faces, points)

        # Also remove from stabilities
        stabilities.pop(min_edge, None)
        edge_to_faces.pop(min_edge, None)

    return sorted(active_edges)


def shapegraph_to_dict(G: nx.Graph) -> dict:
    """Convert a shape graph to a JSON-serializable dictionary."""
    nodes = []
    for nid, attrs in G.nodes(data=True):
        nodes.append({
            "person_id": nid,
            "x": attrs.get("x", 0),
            "y": attrs.get("y", 0),
            "team": attrs.get("team", ""),
            "inferred_role": attrs.get("inferred_role", ""),
            "has_ball": attrs.get("has_ball", False),
            "original_position": attrs.get("original_position", ""),
            "shirt": attrs.get("shirt", 0),
            "name": attrs.get("name", ""),
        })
    edges = []
    for u, v, attrs in G.edges(data=True):
        edges.append({
            "source": u,
            "target": v,
            "distance": attrs.get("distance", 0),
            "cross_team": attrs.get("cross_team", False),
        })
    return {
        "frame_number": G.graph.get("frame_number", 0),
        "timestamp": G.graph.get("timestamp", ""),
        "nodes": nodes,
        "edges": edges,
        "ball": G.graph.get("ball", None),
    }


def save_shapegraphs(results: Dict[int, nx.Graph], path: str):
    """Save shape graphs to a pickle file."""
    with open(path, "wb") as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved {len(results)} shape graphs to {path}")


def save_shapegraphs_json(results: Dict[int, nx.Graph], path: str):
    """Save shape graphs to a JSON file (one graph per line, JSONL format)."""
    import json
    with open(path, "w") as f:
        for fn in sorted(results.keys()):
            d = shapegraph_to_dict(results[fn])
            f.write(json.dumps(d) + "\n")
    print(f"Saved {len(results)} shape graphs to {path}")

