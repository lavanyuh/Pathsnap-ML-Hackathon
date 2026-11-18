"""Raw-file loading for PathSnap trips.

Verified file facts (see EDA): whitespace/tab separated, no headers, CRLF line endings.
  .track  : lon, lat, timestamp_s
  .nodes  : lon, lat            (row index = node_id)
  .arcs   : from_node, to_node  (row index = arc_id)
  .route  : arc_id per line     (reference route, chronological)
"""
from dataclasses import dataclass
from pathlib import Path
import numpy as np


@dataclass
class Trip:
    trip_id: str
    split: str
    track: np.ndarray            # (n, 3) lon, lat, t
    nodes: np.ndarray            # (m, 2) lon, lat
    arcs: np.ndarray             # (k, 2) int from_node, to_node
    route: np.ndarray | None     # (r,) int arc ids, or None if not available


def list_trips(data_root, split):
    """Sorted trip ids inside <data_root>/<split>/."""
    p = Path(data_root) / split
    return sorted(d.name for d in p.iterdir() if d.is_dir())


def load_trip(data_root, split, trip_id, with_route=True):
    base = Path(data_root) / split / trip_id / trip_id
    # np.loadtxt with default delimiter splits on any whitespace and copes with CRLF
    track = np.loadtxt(f"{base}.track", ndmin=2)
    nodes = np.loadtxt(f"{base}.nodes", ndmin=2)
    arcs = np.loadtxt(f"{base}.arcs", dtype=np.int64, ndmin=2)
    route = None
    rp = Path(f"{base}.route")
    if with_route and rp.exists():
        route = np.loadtxt(rp, dtype=np.int64, ndmin=1)
    return Trip(trip_id, split, track, nodes, arcs, route)
