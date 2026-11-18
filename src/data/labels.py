"""Step 3: derive per-point reference labels from the arc-level `.route`.

Problem: `.route` lists the ARCS the vehicle travelled (~390 per trip) but there are ~2,600 GPS points,
so no point has a label. We must assign each point to the arc it was on.

Method (document and defend this in the viva):
  * The vehicle moves forward along the route, so assign points to POSITIONS in the route sequence
    with a monotone (non-decreasing) constraint. Using positions (not arc ids) handles routes that
    revisit the same arc.
  * Choose the assignment minimising total point-to-arc distance (dynamic programming).
  * Distances are TRUNCATED at `cap` metres inside the cost, so points that are far from the route
    (GPS jumps, route truncated) cannot drag their neighbours' assignments around.
  * A point is `labelable` only if its distance to its assigned arc is <= `max_label_dist`.
    Farther points get no label (we do not guess).

Honest caveat: these are DERIVED reference labels, not verified ground truth. The reference route may
be incomplete (see trips 00000002 / 00000033 / 00000054) and the assignment is itself a model.
Note the interaction with candidate generation: labelable points are, by construction, within
`max_label_dist` of their true arc, so a candidate radius >= max_label_dist has miss rate 0 on them.
Therefore choose max_label_dist (trust in labels) and the candidate radius R separately and report both.
"""
import numpy as np


def _dist_to_segment(P, a, b):
    ab = b - a
    L2 = float(ab @ ab)
    if L2 == 0:
        return np.hypot(*(P - a).T)
    t = np.clip(((P - a) @ ab) / L2, 0, 1)
    return np.hypot(*(P - (a + t[:, None] * ab)).T)


def derive_point_labels(net, P, route, max_label_dist=100.0, cap=100.0):
    """net: RoadNetwork, P: (n,2) projected points, route: (m,) arc ids in travel order.

    Returns dict with arrays (length n): route_pos, true_arc_id, assign_dist_m, labelable;
    plus 'n_unassigned_route_arcs'.
    """
    n, m = len(P), len(route)
    D = np.empty((n, m), dtype=np.float32)               # point -> each route position's arc
    for j, arc in enumerate(route):
        D[:, j] = _dist_to_segment(P, net.A[arc], net.B[arc])
    back = np.empty((n, m), dtype=np.int16 if m < 32767 else np.int32)
    ar = np.arange(m)
    cost = np.minimum(D[0], cap).astype(np.float64)
    for i in range(1, n):
        pm = np.minimum.accumulate(cost)                  # best cost over positions <= j
        back[i] = np.maximum.accumulate(np.where(cost <= pm, ar, 0))   # where that best was reached
        cost = np.minimum(D[i], cap) + pm
    pos = np.empty(n, dtype=np.int64)
    pos[-1] = int(np.argmin(cost))
    for i in range(n - 1, 0, -1):
        pos[i - 1] = back[i, pos[i]]
    dist = D[np.arange(n), pos].astype(np.float64)
    return dict(route_pos=pos, true_arc_id=np.asarray(route)[pos], assign_dist_m=dist,
                labelable=dist <= max_label_dist,
                n_unassigned_route_arcs=int(m - len(np.unique(pos))))
