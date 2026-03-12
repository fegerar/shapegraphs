"""shapegraphs — Shape graph computation and analysis for soccer tracking data.

Public API
----------
Computation
  compute_delaunay_edges  : Delaunay edges for a 2-D point set
  compute_shape_graph     : angularly-stable subgraph (Brandes et al. 2025)

Inference
  infer_positions_all     : infer tactical roles for both teams
  infer_positions_for_team: infer tactical roles for one team
  get_nominal_position    : canonical pitch coordinates for a given role
  POSITION_MATRIX         : dict mapping (v_level, h_level) → role string

Frame conversion
  frame_to_shapegraph     : convert a single frame dict → (G, G_nominal)
  generate_shapegraphs    : batch-process pre-parsed frames

I/O helpers
  shapegraph_to_dict      : NetworkX graph → JSON-serialisable dict
  save_shapegraphs        : serialise to pickle
  save_shapegraphs_json   : serialise to JSONL

Visualisation
  plot_shapegraph         : render a shape graph on a matplotlib Axes
"""

from shapegraphs.utils import (
    compute_delaunay_edges,
    compute_shape_graph,
    shapegraph_to_dict,
    save_shapegraphs,
    save_shapegraphs_json,
)
from shapegraphs.inference import (
    infer_positions_all,
    infer_positions_for_team,
    get_nominal_position,
    POSITION_MATRIX,
)
from shapegraphs.frame2sg import (
    frame_to_shapegraph,
    generate_shapegraphs,
)
from shapegraphs.visualize import plot_shapegraph, visualize

__version__ = "0.1.0"

__all__ = [
    # computation
    "compute_delaunay_edges",
    "compute_shape_graph",
    # inference
    "infer_positions_all",
    "infer_positions_for_team",
    "get_nominal_position",
    "POSITION_MATRIX",
    # frame conversion
    "frame_to_shapegraph",
    "generate_shapegraphs",
    # I/O
    "shapegraph_to_dict",
    "save_shapegraphs",
    "save_shapegraphs_json",
    # visualisation
    "plot_shapegraph",
    "visualize",
]
