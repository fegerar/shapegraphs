# Vibecoded by Antigravity

import math
import pickle
import argparse
import os
import sys
from collections import defaultdict

import matplotlib.pyplot as plt
import networkx as nx

def plot_shapegraph(G, ax=None, title=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 7))

    # Pitch dimensions (approximate)
    ax.set_xlim(-55, 55)
    ax.set_ylim(-35, 35)

    # Draw pitch outline
    ax.plot([-52.5, 52.5, 52.5, -52.5, -52.5], [-34, -34, 34, 34, -34], color='black')
    ax.plot([0, 0], [-34, 34], color='black')
    center_circle = plt.Circle((0, 0), 9.15, color='black', fill=False)
    ax.add_patch(center_circle)

    pos = {}
    node_colors = []
    labels = {}

    home_color = 'red'
    away_color = 'blue'

    has_ball_node = None

    for node, data in G.nodes(data=True):
        x = data.get('x', 0)
        y = data.get('y', 0)
        pos[node] = (x, y)
        team = data.get('team', 'unknown')
        if team == 'home':
            node_colors.append(home_color)
        else:
            node_colors.append(away_color)
        
        shirt = data.get('shirt', '')
        role = data.get('inferred_role', '?')
        # labels[node] = f"{shirt}\n{role}"
        labels[node] = str(role)

        if data.get('has_ball', False):
            has_ball_node = node

    # Resolve overlapping nodes by spreading coincident positions in a small circle.
    # Group nodes that share the same (rounded) coordinate.
    _JITTER_RADIUS = 1.5  # metres
    bucket: dict = defaultdict(list)
    for node, (x, y) in pos.items():
        key = (round(x, 3), round(y, 3))
        bucket[key].append(node)

    for key, nodes in bucket.items():
        if len(nodes) == 1:
            continue
        cx, cy = key
        for k, node in enumerate(nodes):
            angle = 2 * math.pi * k / len(nodes)
            pos[node] = (cx + _JITTER_RADIUS * math.cos(angle),
                         cy + _JITTER_RADIUS * math.sin(angle))

    # Draw edges
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.5, edge_color='gray')
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors, node_size=500, alpha=0.8)
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, labels, ax=ax, font_size=8, font_color='white', font_weight='bold')

    # Highlight ball
    if has_ball_node is not None:
        ball_x, ball_y = pos[has_ball_node]
        ax.plot(ball_x, ball_y, 'o', color='yellow', markersize=8, markeredgecolor='black', label='Ball Carrier')

    ax.set_aspect('equal')
    if title:
        ax.set_title(title)
        
    ax.axis('off')

def visualize(
    data: dict,
    frame: int = None,
    output: str = "shapegraph_viz.png",
    video: bool = False,
    fps: int = 10,
):
    """
    Visualize shape graphs from a loaded data dict.

    Parameters
    ----------
    data
        Dict mapping frame number → graph entry, as returned by
        ``generate_shapegraphs`` or loaded from a pickle file.
        Each entry is either a NetworkX graph or a dict with
        ``"original"`` and ``"nominal"`` keys.
    frame
        Frame number to render (single-image mode only).
        Defaults to the first available frame.
    output
        Output file path. Use a ``.png`` extension for images and
        ``.mp4`` for videos. In video mode the extension is forced
        to ``.mp4`` automatically.
    video
        If True, render all frames as a video instead of a single image.
    fps
        Frames per second for video output.
    """
    frames = sorted(data.keys())

    if video:
        import io
        import cv2
        import numpy as np

        out_path = output
        if not out_path.endswith(".mp4"):
            out_path = (out_path.rsplit(".", 1)[0] if "." in out_path else out_path) + ".mp4"

        print(f"Generating video with {len(frames)} frames at {fps} FPS...")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = None

        sample = data[frames[0]]
        has_nominal = isinstance(sample, dict) and "nominal" in sample

        if has_nominal:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 8))
        else:
            fig, ax = plt.subplots(figsize=(12, 8))

        for i, frame_id in enumerate(frames):
            entry = data[frame_id]
            if has_nominal:
                ax1.clear()
                ax2.clear()
                plot_shapegraph(entry["original"], ax=ax1, title=f"Actual positions — Frame {frame_id}")
                plot_shapegraph(entry["nominal"],  ax=ax2, title=f"Nominal positions — Frame {frame_id}")
            else:
                ax.clear()
                G = entry if not isinstance(entry, dict) else entry.get("original", entry)
                plot_shapegraph(G, ax=ax, title=f"Shapegraph — Frame {frame_id}")

            plt.tight_layout()
            buf = io.BytesIO()
            plt.savefig(buf, format="png", dpi=150)
            buf.seek(0)
            img = cv2.imdecode(np.frombuffer(buf.getvalue(), dtype=np.uint8), cv2.IMREAD_COLOR)

            if writer is None:
                h, w, _ = img.shape
                writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
            writer.write(img)

            if i > 0 and i % 10 == 0:
                print(f"Processed {i}/{len(frames)} frames...")

        if writer is not None:
            writer.release()
        plt.close(fig)
        print(f"Saved video to {out_path}")

    else:
        target_frame = frame if frame is not None else frames[0]
        if target_frame not in data:
            print(f"Frame {target_frame} not found, falling back to {frames[0]}.")
            target_frame = frames[0]

        entry = data[target_frame]
        has_nominal = isinstance(entry, dict) and "nominal" in entry

        if has_nominal:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 8))
            plot_shapegraph(entry["original"], ax=ax1, title=f"Actual positions — Frame {target_frame}")
            plot_shapegraph(entry["nominal"],  ax=ax2, title=f"Nominal positions — Frame {target_frame}")
        else:
            G = entry if not isinstance(entry, dict) else entry.get("original", entry)
            fig, ax = plt.subplots(figsize=(12, 8))
            plot_shapegraph(G, ax=ax, title=f"Shapegraph — Frame {target_frame}")

        plt.tight_layout()
        plt.savefig(output, dpi=150)
        print(f"Saved visualization to {output}")

