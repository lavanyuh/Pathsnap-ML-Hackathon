"""Per-trip projected road network with a spatial index.

Design decisions (document these in the report):
  * Arc IDs are NEVER renumbered. `.route` refers to arcs by row index, so deleting
    degenerate arcs would silently corrupt every label. Instead we mask arcs
    (`valid`) and keep the original IDs everywhere.
  * Degenerate arcs (zero length, self-loops) are marked invalid and never become candidates.
  * Duplicate (parallel) arcs are kept: the reference route may use either copy.
  * Arcs are directed. `has_reverse` records whether the opposite direction exists
    (False => one-way road); the HMM and heading features use this later.
  * Projection is per trip (trips are in different cities), local equirectangular, metres.
"""
from dataclasses import dataclass
import numpy as np
import shapely
from shapely import STRtree
from .geo import local_projector, point_segment_distance


@dataclass
class RoadNetwork:
    trip_id: str
    to_xy: callable
    node_xy: np.ndarray      # (m,2) metres
    arcs: np.ndarray         # (k,2) original from/to node ids
    A: np.ndarray            # (k,2) arc start xy
    B: np.ndarray            # (k,2) arc end xy
    length: np.ndarray       # (k,) metres
    bearing: np.ndarray      # (k,) radians, atan2(dy,dx) in the projected frame
    valid: np.ndarray        # (k,) bool: not zero-length, not a self-loop
    has_reverse: np.ndarray  # (k,) bool
    out_arcs: list           # out_arcs[node] -> np.array of arc ids leaving node
    in_arcs: list
    _tree: STRtree = None
    _tree_ids: np.ndarray = None   # tree position -> original arc id

    # ------------------------------------------------------------------ build
    @classmethod
    def from_trip(cls, trip, origin=None):
        lon0, lat0 = origin if origin is not None else trip.track[:, :2].mean(0)
        to_xy = local_projector(lon0, lat0)
        N = to_xy(trip.nodes[:, 0], trip.nodes[:, 1])
        ar = trip.arcs
        A, B = N[ar[:, 0]], N[ar[:, 1]]
        d = B - A
        length = np.hypot(d[:, 0], d[:, 1])
        valid = (length > 0) & (ar[:, 0] != ar[:, 1])
        pairs = set(map(tuple, ar.tolist()))
        has_rev = np.fromiter(((b, a) in pairs for a, b in ar.tolist()), bool, len(ar))
        m = len(N)
        out_arcs = [[] for _ in range(m)]; in_arcs = [[] for _ in range(m)]
        for k in np.flatnonzero(valid):
            out_arcs[ar[k, 0]].append(k); in_arcs[ar[k, 1]].append(k)
        out_arcs = [np.array(x, dtype=np.int64) for x in out_arcs]
        in_arcs = [np.array(x, dtype=np.int64) for x in in_arcs]
        net = cls(trip.trip_id, to_xy, N, ar, A, B, length, np.arctan2(d[:, 1], d[:, 0]),
                  valid, has_rev, out_arcs, in_arcs)
        ids = np.flatnonzero(valid)
        net._tree_ids = ids
        net._tree = STRtree(shapely.linestrings(np.stack([A[ids], B[ids]], axis=1)))
        return net

    # ------------------------------------------------------------- candidates
    def arcs_within(self, P, radius):
        """All valid arcs within `radius` metres of each point.

        Returns (pt_idx, arc_id, dist, t): parallel arrays, one row per (point, candidate)
        pair; `t` in [0,1] is where the point projects along the arc.
        """
        pi, ti = self._tree.query(shapely.points(P), predicate="dwithin", distance=radius)
        arc = self._tree_ids[ti]
        dist, t = point_segment_distance(P[pi], self.A[arc], self.B[arc])
        keep = dist <= radius
        return pi[keep], arc[keep], dist[keep], t[keep]


def prepare_track(track, to_xy, gap_s=5.0, max_speed_ms=70.0):
    """Project GPS points and derive per-step quantities. NEVER drops points: the API
    must return one prediction per observation.

    Flags (not deletions):
      gap_before : time since previous point exceeds `gap_s`
      speed_flag : implied speed above `max_speed_ms` (~250 km/h) -> likely GPS jump
    """
    xy = to_xy(track[:, 0], track[:, 1])
    t = track[:, 2]
    dt = np.r_[np.nan, np.diff(t)]
    step = np.r_[np.nan, np.hypot(*np.diff(xy, axis=0).T)]
    speed = step / dt
    return dict(xy=xy, t=t, dt=dt, step=step, speed=speed,
                gap_before=np.nan_to_num(dt, nan=0) > gap_s,
                speed_flag=np.nan_to_num(speed, nan=0) > max_speed_ms)
