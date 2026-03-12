"""
Bassek / DFL XML reader for shapegraphs.

Usage
-----
::

    from shapegraphs.readers.bassek import generate_shapegraphs_from_files

    results = generate_shapegraphs_from_files(
        match_info_path="MatchParameters.xml",
        position_data_path="PositionData.xml",
    )
    # results: Dict[frame_num, {"original": G, "nominal": G_nominal}]
"""

import xml.etree.ElementTree as ET
from typing import Dict, Optional, Tuple

from shapegraphs.frame2sg import generate_shapegraphs


def parse_match_info(path: str) -> dict:
    """
    Parse the match information XML file (Bassek / DFL format).

    Returns a dict with keys:
      - "home_team_id", "away_team_id"
      - "home_team_name", "away_team_name"
      - "pitch_x", "pitch_y"  (pitch dimensions in metres)
      - "players": {person_id: {"team": "home"|"away",
                                "position": str,
                                "name": str,
                                "shirt": int,
                                "starting": bool}}
    """
    tree = ET.parse(path)
    root = tree.getroot()

    match_info_el = root.find("MatchInformation")
    if match_info_el is not None:
        root = match_info_el

    info: dict = {"players": {}}

    gen = root.find("General")
    if gen is not None:
        info["home_team_id"] = gen.get("HomeTeamId", "")
        info["away_team_id"] = gen.get("GuestTeamId", "")
        info["home_team_name"] = gen.get("HomeTeamName", "")
        info["away_team_name"] = gen.get("GuestTeamName", "")

    env = root.find("Environment")
    if env is not None:
        info["pitch_x"] = float(env.get("PitchX", "105.0"))
        info["pitch_y"] = float(env.get("PitchY", "68.0"))
    else:
        info["pitch_x"] = 105.0
        info["pitch_y"] = 68.0

    teams_el = root.find("Teams")
    for team_el in teams_el.findall("Team"):
        role = team_el.get("Role", "").lower()
        team_label = "home" if role == "home" else "away"

        players_el = team_el.find("Players")
        if players_el is None:
            continue
        for p in players_el.findall("Player"):
            pid = p.get("PersonId", "")
            info["players"][pid] = {
                "team": team_label,
                "position": p.get("PlayingPosition", ""),
                "name": f'{p.get("FirstName", "")} {p.get("LastName", "")}',
                "shirt": int(p.get("ShirtNumber", "0")),
                "starting": p.get("Starting", "false").lower() == "true",
            }

    return info


def parse_position_data(
    path: str,
    match_info: dict,
    frame_range: Optional[Tuple[int, int]] = None,
) -> dict:
    """
    Parse the position data XML (streaming parser for memory efficiency).

    Returns a dict mapping frame number → {
        "players": {person_id: {"x", "y", "team", "speed"}},
        "ball":    {"x", "y", "z", "possession", "status"} or None,
        "timestamp": str,
    }
    """
    frames: dict = {}

    home_tid = match_info.get("home_team_id", "")
    away_tid = match_info.get("away_team_id", "")

    context = ET.iterparse(path, events=("start", "end"))
    current_person_id = None
    current_team_id = None
    in_ball = False

    for event, elem in context:
        if event == "start" and elem.tag == "FrameSet":
            current_person_id = elem.get("PersonId", "")
            current_team_id = elem.get("TeamId", "")
            in_ball = (
                current_person_id.upper() == "BALL"
                or "BALL" in elem.get("TeamId", "").upper()
            )

        elif event == "end" and elem.tag == "Frame":
            n = int(elem.get("N", "0"))

            if frame_range is not None:
                if n < frame_range[0] or n >= frame_range[1]:
                    elem.clear()
                    continue

            x = float(elem.get("X", "0"))
            y = float(elem.get("Y", "0"))
            t = elem.get("T", "")

            if n not in frames:
                frames[n] = {"players": {}, "ball": None, "timestamp": t}

            if in_ball:
                z = float(elem.get("Z", "0"))
                bp = int(elem.get("BallPossession", "0"))
                bs = int(elem.get("BallStatus", "0"))
                frames[n]["ball"] = {
                    "x": x, "y": y, "z": z,
                    "possession": bp, "status": bs,
                }
            else:
                if current_team_id == home_tid:
                    team_label = "home"
                elif current_team_id == away_tid:
                    team_label = "away"
                else:
                    pinfo = match_info["players"].get(current_person_id, {})
                    team_label = pinfo.get("team", "unknown")

                s = float(elem.get("S", "0"))
                frames[n]["players"][current_person_id] = {
                    "x": x, "y": y, "team": team_label, "speed": s,
                }

            if not frames[n]["timestamp"]:
                frames[n]["timestamp"] = t

            elem.clear()

        elif event == "end" and elem.tag == "FrameSet":
            current_person_id = None
            current_team_id = None
            in_ball = False
            elem.clear()

    return frames

# Frame number below which we consider a frame to belong to the first half.
# This is a heuristic specific to the Bassek / DFL dataset.
_HALFBREAK_FRAME = 80_000


def _detect_attacking_direction(frames: dict, match_info: dict) -> bool:
    """
    Heuristic: compare X positions of both goalkeepers in the first available
    frame.  The team whose GK has the smaller X coordinate is defending the
    left goal, therefore attacking toward +X ("up").

    Returns True if the home team attacks "up" (+X) in the first half.
    The position code "TW" (Torwart) is used because it is the DFL standard.
    """
    if not frames or not match_info.get("players"):
        return True

    first_frame = frames[min(frames)]
    players_data = first_frame.get("players", {})

    home_gk_x: Optional[float] = None
    away_gk_x: Optional[float] = None

    for pid, pd in players_data.items():
        pinfo = match_info["players"].get(pid, {})
        if pinfo.get("position", "") == "TW":  # DFL goalkeeper code
            if pd["team"] == "home":
                home_gk_x = pd["x"]
            elif pd["team"] == "away":
                away_gk_x = pd["x"]

    if home_gk_x is not None and away_gk_x is not None:
        return home_gk_x < away_gk_x
    return True


def generate_shapegraphs_from_files(
    match_info_path: str,
    position_data_path: str,
    frame_range: Optional[Tuple[int, int]] = None,
    ball_in_play_only: bool = True,
    halfbreak_frame: int = _HALFBREAK_FRAME,
    verbose: bool = True,
) -> Dict[int, dict]:
    """
    Parse Bassek / DFL XML files and return shape graphs for each frame.

    Parameters
    ----------
    match_info_path
        Path to the MatchParameters XML file.
    position_data_path
        Path to the PositionData XML file.
    frame_range
        Optional ``(start, end)`` tuple to limit which frames are processed.
    ball_in_play_only
        If True (default), frames where ``ball["status"] == 0`` are skipped.
    halfbreak_frame
        Frame number that separates the first and second halves.
        Defaults to 80 000 (Bassek dataset convention).
    verbose
        Print progress information.

    Returns
    -------
    Dict[int, dict]
        Mapping of frame number → ``{"original": G, "nominal": G_nominal}``.
    """
    if verbose:
        print("Parsing match information...")
    match_info = parse_match_info(match_info_path)
    if verbose:
        n_players = len(match_info.get("players", {}))
        print(f"  Found {n_players} players in match info")

    if verbose:
        print("Parsing position data...")
    frames = parse_position_data(position_data_path, match_info, frame_range)
    if verbose:
        print(f"  Loaded {len(frames)} frames")

    home_attacking_up = _detect_attacking_direction(frames, match_info)

    return generate_shapegraphs(
        frames=frames,
        match_info=match_info,
        home_attacking_up_first_half=home_attacking_up,
        halfbreak_frame=halfbreak_frame,
        ball_in_play_only=ball_in_play_only,
        verbose=verbose,
    )
