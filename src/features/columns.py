"""Single source of truth for feature names. Everyone imports FEATURE_COLS from here.

Rules (document in the report): no absolute lon/lat, no trip id, nothing derived from labels or `.route`.
Features describe the geometry/kinematics of a (GPS point, candidate arc) pair.
"""
KEY_COLS = ["trip_id", "point_idx", "arc_id"]
LABEL_COLS = ["label", "labelable"]       # label: 1/0 for labelable rows, -1 when the point is not labelable

FEATURE_COLS = [
    # --- geometry of the pair
    "dist_m",          # perpendicular distance point -> arc segment (m)
    "dist_rel_min",    # dist_m minus the smallest candidate distance of that point (m)
    "dist_rank",       # 0 = nearest candidate of the point
    "proj_t",          # where the point projects along the arc (0 = start, 1 = end)
    "along_m",         # proj_t * arc length (m from arc start)
    "rem_m",           # (1 - proj_t) * arc length (m to arc end)
    "arc_len_m",       # arc length (m)
    # --- direction
    "heading_cos",     # cos(angle between GPS motion direction and arc bearing); 0 if heading unreliable
    "heading_valid",   # 1 if the vehicle moved enough (>=3 m over +-2 points, no gap) for a heading
    "is_oneway",       # 1 if the arc has no reverse arc
    "turn_angle",      # |heading change| around the point (rad), 0 if unreliable
    # --- topology
    "deg_from", "deg_to",   # total degree of the arc's start / end node (junction-ness)
    "n_candidates",    # candidates for this point
    # --- kinematics / timing
    "speed_ms",        # displacement over +-2 points / time span (0 if a gap makes it unreliable)
    "dt_s",            # seconds since previous point, clipped to 60
    "gap_before",      # 1 if dt > 5 s
    # --- trajectory context: how well do NEIGHBOURING points agree with this arc?
    "win3_mean_dist", "win3_max_dist",   # over points within +-3 steps (m)
    "win8_mean_dist",                    # over points within +-8 steps (m)
]
