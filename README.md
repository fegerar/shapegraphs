# shapegraphs

A Python library for computing, analysing, and visualising **shape graphs** of soccer tracking data.

This library is a custom implementation of the shape graph framework introduced by:

> Ulrik Brandes, Hadi Sotudeh, Doğan Parlak, Paolo Laffranchi & Mert Erkul  (2025).  
> *Shape graphs and the instantaneous inference of tactical positions in soccer*  
> npj Complexity. https://doi.org/10.1038/s44260-025-00047-x

Shape graphs are planar subgraphs of the Delaunay triangulation of player positions, constructed by iteratively removing edges whose angular stability falls below a threshold. They provide a compact, interpretable representation of team spatial structure and formation.

---

## Installation

Clone the repository and install in editable mode from the `shapegraphs/` directory:

```bash
pip install git+https://github.com/fegerar/shapegraphs.git
```

---

## Quick start

### From raw tracking data (Bassek / DFL XML format)

```python
from shapegraphs.readers.bassek import (
    parse_match_info,
    parse_position_data,
    generate_shapegraphs_from_files,
)

results = generate_shapegraphs_from_files(
    match_info_path="MatchParameters.xml",
    position_data_path="PositionData.xml",
)
# results: Dict[frame_num, {"original": G, "nominal": G_nominal}]
```

### From pre-parsed frames

```python
import shapegraphs as sg

# parse your data however you like, then:
results = sg.generate_shapegraphs(frames, match_info)
```

Each entry in `results` contains two NetworkX graphs:
- **`"original"`** — shape graph built from actual player positions.
- **`"nominal"`** — shape graph built from the ideal formation positions inferred for that frame.

### Core algorithm

```python
import numpy as np
from shapegraphs import compute_shape_graph

points = np.array([...])  # (n, 2) player coordinates
edges = compute_shape_graph(points)  # list of (i, j) index pairs
```

### Infer tactical roles

```python
from shapegraphs import infer_positions_all

positions = infer_positions_all(
    points,
    home_indices=[0, 1, ..., 10],
    away_indices=[11, 12, ..., 21],
    home_attacking_up=True,
)
# {player_index: "CB" | "CM" | "GK" | ...}
```

### Visualisation

```python
from shapegraphs import visualize

# Single frame → PNG
visualize(results, frame=1234, output="frame.png")

# All frames → MP4
visualize(results, video=True, fps=25, output="match.mp4")
```

Or render directly onto a matplotlib `Axes`:

```python
import matplotlib.pyplot as plt
from shapegraphs import plot_shapegraph

fig, ax = plt.subplots(figsize=(12, 8))
plot_shapegraph(results[1234]["original"], ax=ax, title="Frame 1234")
plt.show()
```

### Save / load

```python
from shapegraphs import save_shapegraphs, save_shapegraphs_json

save_shapegraphs(results, "shapegraphs.pkl")       # pickle
save_shapegraphs_json(results, "shapegraphs.jsonl") # JSONL
```
---

## Reference

Ulrik Brandes, Hadi Sotudeh, Doğan Parlak, Paolo Laffranchi & Mert Erkul (2025). *Shape graphs and the instantaneous inference of tactical positions in soccer* npj Complexity. https://doi.org/10.1038/s44260-025-00047-x
