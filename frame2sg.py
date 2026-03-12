"""
Frame to Shape Graph
converting each frame of soccer tracking data into a shape graph representation.

The functions here are data-source agnostic: they accept pre-parsed Python
dicts rather than any specific file format.  For Bassek / DFL-specific I/O
(reading XML files, detecting halftime from goalkeeper positions, etc.) see
``shapegraphs.readers.bassek``.
"""

import math
from typing import Dict, Optional, Tuple
import networkx as nx
import numpy as np
from tqdm import tqdm

from shapegraphs.inference import infer_positions_all, get_nominal_position
from shapegraphs.utils import compute_shape_graph


def frame_to_shapegraph(
    frame_data: dict,
    match_info: dict,
    home_attacking_up_first_half: bool = True,
    game_section: str = "firstHalf",
) -> Optional[Tuple[nx.Graph, nx.Graph]]:
    """
    Convert a single frame of tracking data to a shape graph (NetworkX graph).
    """
    players_data = frame_data.get("players", {})
    ball_data = frame_data.get("ball", None)

    if len(players_data) < 4:
        return None, None

    # Build ordered arrays
    # Filter out referees and unknown persons
    person_ids = [pid for pid in players_data.keys() if players_data[pid].get("team") in ("home", "away")]
    n = len(person_ids)
    points = np.zeros((n, 2))
    teams = []

    for i, pid in enumerate(person_ids):
        pd = players_data[pid]
        points[i] = [pd["x"], pd["y"]]
        teams.append(pd["team"])

    home_indices = [i for i, t in enumerate(teams) if t == "home"]
    away_indices = [i for i, t in enumerate(teams) if t == "away"]

    # Compute shape graph on ALL players (22 nodes)
    sg_edges = compute_shape_graph(points, alpha_threshold=math.pi / 4)

    # Determine attacking direction
    if game_section == "firstHalf":
        home_up = home_attacking_up_first_half
    else:
        home_up = not home_attacking_up_first_half

    # Infer positions
    positions = infer_positions_all(
        points, home_indices, away_indices, home_attacking_up=home_up)

    # Determine who has the ball
    has_ball_idx = None
    if ball_data is not None:
        ball_pos = np.array([ball_data["x"], ball_data["y"]])
        dists = np.linalg.norm(points - ball_pos, axis=1)
        has_ball_idx = int(np.argmin(dists))

    # Build NetworkX graph
    G = nx.Graph()
    for i, pid in enumerate(person_ids):
        pinfo = match_info["players"].get(pid, {})
        G.add_node(pid,
                   index=i,
                   x=float(points[i, 0]),
                   y=float(points[i, 1]),
                   team=teams[i],
                   inferred_role=positions.get(i, "?"),
                   original_position=pinfo.get("position", ""),
                   shirt=pinfo.get("shirt", 0),
                   name=pinfo.get("name", ""),
                   has_ball=(i == has_ball_idx)
                )

    for u, v in sg_edges:
        pid_u = person_ids[u]
        pid_v = person_ids[v]
        dist = float(np.linalg.norm(points[u] - points[v]))
        cross_team = teams[u] != teams[v]
        G.add_edge(pid_u, pid_v, distance=dist, cross_team=cross_team)

    G.graph["timestamp"] = frame_data.get("timestamp", "")
    G.graph["ball"] = ball_data

    nominal_points = np.zeros((n, 2))
    for i in range(n):
        role = positions.get(i, "CM")
        atk_dir = "up" if (teams[i] == "home") == home_up else "down"
        nom_x, nom_y = get_nominal_position(role, atk_dir)
        nominal_points[i] = [nom_x, nom_y]

    sg_edges_nominal = compute_shape_graph(nominal_points, alpha_threshold=math.pi / 4)

    G_nominal = nx.Graph()
    for i, pid in enumerate(person_ids):
        pinfo = match_info["players"].get(pid, {})
        G_nominal.add_node(pid,
                           index=i,
                           x=float(nominal_points[i, 0]),
                           y=float(nominal_points[i, 1]),
                           team=teams[i],
                           inferred_role=positions.get(i, "?"),
                           original_position=pinfo.get("position", ""),
                           shirt=pinfo.get("shirt", 0),
                           name=pinfo.get("name", ""),
                           has_ball=(i == has_ball_idx)
                        )

    for u, v in sg_edges_nominal:
        pid_u = person_ids[u]
        pid_v = person_ids[v]
        dist = float(np.linalg.norm(nominal_points[u] - nominal_points[v]))
        cross_team = teams[u] != teams[v]
        G_nominal.add_edge(pid_u, pid_v, distance=dist, cross_team=cross_team)

    G_nominal.graph["timestamp"] = frame_data.get("timestamp", "")
    G_nominal.graph["ball"] = ball_data

    return G, G_nominal


def generate_shapegraphs(
    frames: Dict[int, dict],
    match_info: dict,
    home_attacking_up_first_half: bool = True,
    halfbreak_frame: Optional[int] = None,
    ball_in_play_only: bool = True,
    verbose: bool = True,
) -> Dict[int, dict]:
    """
    Generate shape graphs for a collection of pre-parsed tracking frames.

    Parameters
    ----------
    frames
        Mapping of frame number → frame dict.  Each frame dict must contain:
        - ``"players"``: dict of player_id → {"x", "y", "team" ("home"/"away")}
        - ``"ball"`` (optional): {"x", "y", "status"}
        - ``"timestamp"`` (optional): str
    match_info
        Dict with at least a ``"players"`` key mapping player_id →
        {"position", "shirt", "name"}.
    home_attacking_up_first_half
        Whether the home team attacks in the +X direction during the first half.
    halfbreak_frame
        Frame number where the second half begins.  Frames with a number
        *strictly less than* this value are treated as first-half frames.
        Pass ``None`` to treat every frame as a first-half frame.
    ball_in_play_only
        Skip frames where ``ball["status"] == 0``.
    verbose
        Print progress information.

    Returns
    -------
    Dict[int, dict]
        Mapping of frame number → {``"original"``: G, ``"nominal"``: G_nominal}.
    """
    results: Dict[int, dict] = {}
    frame_numbers = sorted(frames.keys())

    for fn in tqdm(frame_numbers, desc="Processing frames", disable=not verbose):
        fd = frames[fn]

        # Skip if ball not in play
        if ball_in_play_only and fd.get("ball") is not None:
            if fd["ball"].get("status", 1) == 0:
                continue

        if halfbreak_frame is None:
            game_section = "firstHalf"
        else:
            game_section = "firstHalf" if fn < halfbreak_frame else "secondHalf"

        G, G_nominal = frame_to_shapegraph(
            fd, match_info, home_attacking_up_first_half, game_section
        )
        if G is not None:
            G.graph["frame_number"] = fn
            G_nominal.graph["frame_number"] = fn
            results[fn] = {"original": G, "nominal": G_nominal}

    if verbose:
        print(f"Generated {len(results)} shape graphs")

    return results
