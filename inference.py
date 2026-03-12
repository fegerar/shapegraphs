"""
Inference roles from shapegraphs.
"""

import math
from typing import Dict, List, Tuple
import numpy as np

from shapegraphs.utils import _build_face_structure, compute_shape_graph


POSITION_MATRIX = {
    (4, 0): "LF",  (4, 1): "LCF", (4, 2): "CF",  (4, 3): "RCF", (4, 4): "RF",
    (3, 0): "LAM", (3, 1): "LW",  (3, 2): "CAM", (3, 3): "RW",  (3, 4): "RAM",
    (2, 0): "LM",  (2, 1): "LCM", (2, 2): "CM",  (2, 3): "RCM", (2, 4): "RM",
    (1, 0): "LDM", (1, 1): "LCDM",(1, 2): "CDM", (1, 3): "RCDM",(1, 4): "RDM",
    (0, 0): "LB",  (0, 1): "LCB", (0, 2): "CB",  (0, 3): "RCB", (0, 4): "RB",
}

# Canonical (x, y) positions for each role in "attacking up" (+X) coordinates.
# Row (v_level): 0=defenders → 4=forwards along X axis.
# Col (h_level): 0=left (+Y) → 4=right (-Y) from team's perspective.
_NOMINAL_LONG = {0: -35.0, 1: -20.0, 2: 0.0, 3: 20.0, 4: 35.0}
_NOMINAL_TRANS_UP = {0: 25.0, 1: 12.5, 2: 0.0, 3: -12.5, 4: -25.0}

_NOMINAL_POSITIONS_UP: Dict[str, Tuple[float, float]] = {
    role: (_NOMINAL_LONG[r], _NOMINAL_TRANS_UP[c])
    for (r, c), role in POSITION_MATRIX.items()
}
_NOMINAL_POSITIONS_UP["GK"] = (-48.0, 0.0)


def get_nominal_position(role: str, attacking_direction: str = "up") -> Tuple[float, float]:
    """
    Return the nominal (x, y) pitch coordinates for a given role.

    For teams attacking down (-X), the field is mirrored on both axes so
    that the formation stays consistent from each team's perspective.
    """
    x, y = _NOMINAL_POSITIONS_UP.get(role, (0.0, 0.0))
    if attacking_direction != "up":
        x, y = -x, -y
    return (x, y)


def _split_levels(indices: List[int], values: np.ndarray,
                  face_centers: List[float]) -> List[List[int]]:
    """
    Split a set of player indices into up to 5 ordinal levels along one axis
    using the two-tier decomposition from Brandes et al.

    The procedure (following the paper's Fig. 3):
      1. Find the two extreme face centers (min and max).
      2. Use them as horizontal thresholds to split off the top and bottom.
         Players at or below the lowest face center → bottom group.
         Players at or above the highest face center → top group.
      3. Repeat the same procedure once on the remaining middle group,
         using face centers that fall within the middle range.

    This yields up to 5 levels: [bottom, def-mid, mid, att-mid, top].
    """
    if len(indices) <= 1:
        return [indices]

    sorted_idx = sorted(indices, key=lambda i: values[i])

    if len(face_centers) < 2:
        # Not enough faces → equidistant split
        n = len(sorted_idx)
        if n <= 2:
            return [sorted_idx]
        third = max(1, n // 3)
        rem = n - 2 * third
        return [sorted_idx[:third],
                sorted_idx[third:third + rem],
                sorted_idx[third + rem:]]

    fc_sorted = sorted(face_centers)
    fc_lo = fc_sorted[0]    # lowest face center
    fc_hi = fc_sorted[-1]   # highest face center

    # First-tier split: separate bottom and top from middle
    bottom = [i for i in indices if values[i] < fc_lo]
    top = [i for i in indices if values[i] > fc_hi]
    middle = [i for i in indices if fc_lo <= values[i] <= fc_hi]

    # If first tier yields empty bottom/top, pull the extreme player(s)
    if not bottom and middle:
        min_val = min(values[i] for i in middle)
        bottom = [i for i in middle if values[i] == min_val]
        middle = [i for i in middle if values[i] != min_val]
    if not top and middle:
        max_val = max(values[i] for i in middle)
        top = [i for i in middle if values[i] == max_val]
        middle = [i for i in middle if values[i] != max_val]

    # Edge case: everything in one group
    if not bottom and not top:
        bottom = [sorted_idx[0]]
        top = [sorted_idx[-1]]
        middle = sorted_idx[1:-1]

    # Second-tier split on the middle group
    if len(middle) >= 3:
        # Use face centers within the middle range as secondary thresholds
        mid_lo = min(values[i] for i in middle)
        mid_hi = max(values[i] for i in middle)
        mid_fcs = [fc for fc in fc_sorted if mid_lo <= fc <= mid_hi]

        if len(mid_fcs) >= 2:
            mfc_lo = mid_fcs[0]
            mfc_hi = mid_fcs[-1]
        else:
            # Equal thirds fallback
            span = mid_hi - mid_lo
            mfc_lo = mid_lo + span / 3
            mfc_hi = mid_hi - span / 3

        mid_low = [i for i in middle if values[i] < mfc_lo]
        mid_high = [i for i in middle if values[i] > mfc_hi]
        mid_center = [i for i in middle if mfc_lo <= values[i] <= mfc_hi]

        if not mid_center:
            mid_center = middle
            mid_low = []
            mid_high = []

        levels = [bottom, mid_low, mid_center, mid_high, top]
    elif len(middle) == 2:
        ms = sorted(middle, key=lambda i: values[i])
        levels = [bottom, [ms[0]], [], [ms[1]], top]
    elif len(middle) == 1:
        levels = [bottom, [], middle, [], top]
    else:
        levels = [bottom, [], [], [], top]

    # Remove empty levels and return
    levels = [lv for lv in levels if lv]
    return levels


def _compute_internal_face_centers(
    points: np.ndarray,
    sg_edges: List[Tuple[int, int]],
) -> Tuple[List[float], List[float]]:
    """
    Compute centers of internal faces of a shape graph.
    Returns (face_centers_y, face_centers_x).
    """
    if len(points) < 3 or len(sg_edges) < 3:
        return [], []

    try:
        faces, _ = _build_face_structure(sg_edges, points)
    except Exception:
        return [], []

    centers_y = []
    centers_x = []

    for face in faces:
        if len(face) < 3:
            continue
        # Skip the outer (unbounded) face: it's the one with the largest area
        # Heuristic: the outer face typically has the most vertices or wraps around
        # We use the signed area test — the outer face has negative area (CW)
        area = 0.0
        for i, v in enumerate(face):
            v_1 = (i + 1) % len(face)
            area += points[v, 0] * points[face[v_1], 1]
            area -= points[face[v_1], 0] * points[v, 1]
        if area < 0:
            continue  # outer face (clockwise)

        cy = float(np.mean([points[v, 1] for v in face]))
        cx = float(np.mean([points[v, 0] for v in face]))
        centers_y.append(cy)
        centers_x.append(cx)

    return centers_y, centers_x


def infer_positions_for_team(
    player_indices: List[int],
    points: np.ndarray,
    attacking_direction: str = "up",
) -> Dict[int, str]:
    """
    Infer tactical positions for one team's players.
    """
    if not player_indices:
        return {}

    # Identify GK as the player deepest toward own goal
    if attacking_direction == "up":
        gk_idx = min(player_indices, key=lambda i: points[i][0])
    else:
        gk_idx = max(player_indices, key=lambda i: points[i][0])

    outfield_indices = [i for i in player_indices if i != gk_idx]

    if len(outfield_indices) < 2:
        return {gk_idx: "GK", **{i: "CM" for i in outfield_indices}}

    # Build local coordinate array for outfield players
    team_pts = points[outfield_indices]
    n_of = len(outfield_indices)

    # Compute a per-team shape graph on outfield players only
    team_sg_edges = compute_shape_graph(team_pts, alpha_threshold=math.pi / 4)

    # Get face centers from the per-team shape graph
    fc_y, fc_x = _compute_internal_face_centers(team_pts, team_sg_edges)

    if attacking_direction == "up":
        # Attacking +X
        y_vals = team_pts[:, 0].copy()
        fc_long = list(fc_x)
        # Left is +Y, Right is -Y. To make Left the smallest value (index 0), negate Y.
        x_vals = -team_pts[:, 1].copy()
        fc_trans = [-c for c in fc_y]
    else:
        # Attacking -X
        y_vals = -team_pts[:, 0].copy()
        fc_long = [-c for c in fc_x]
        # Left is -Y, Right is +Y. Left is already smallest.
        x_vals = team_pts[:, 1].copy()
        fc_trans = list(fc_y)

    # Vertical split (backs → forwards)
    local_indices = list(range(n_of))
    v_levels = _split_levels(local_indices, y_vals, fc_long)

    # Horizontal split (left → right)
    h_levels = _split_levels(local_indices, x_vals, fc_trans)

    # Map each local player to a normalized 0-4 level
    v_map = {}
    n_vlev = len(v_levels)
    for lv_idx, members in enumerate(v_levels):
        if n_vlev == 1:
            norm_lv = 2  # all in middle
        else:
            norm_lv = int(round(lv_idx * 4 / (n_vlev - 1)))
        for m in members:
            v_map[m] = norm_lv

    h_map = {}
    n_hlev = len(h_levels)
    for lv_idx, members in enumerate(h_levels):
        if n_hlev == 1:
            norm_lv = 2
        else:
            norm_lv = int(round(lv_idx * 4 / (n_hlev - 1)))
        for m in members:
            h_map[m] = norm_lv

    # Assign position labels
    positions = {gk_idx: "GK"}
    for local_idx, global_idx in enumerate(outfield_indices):
        v = v_map.get(local_idx, 2)
        h = h_map.get(local_idx, 2)
        label = POSITION_MATRIX.get((v, h), "CM")
        positions[global_idx] = label

    return positions


def infer_positions_all(
    points: np.ndarray,
    home_indices: List[int],
    away_indices: List[int],
    home_attacking_up: bool = True,
) -> Dict[int, str]:
    """
    Infer tactical positions for both teams.
    GK identification is handled within each team's inference.
    """
    home_dir = "up" if home_attacking_up else "down"
    away_dir = "down" if home_attacking_up else "up"

    pos_home = infer_positions_for_team(home_indices, points, home_dir)
    pos_away = infer_positions_for_team(away_indices, points, away_dir)

    return {**pos_home, **pos_away}
