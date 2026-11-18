"""Geographic helpers. Each trip is in a different city, so we project per trip."""
import numpy as np

EARTH_R = 6371008.8  # mean Earth radius, metres


def haversine_m(lon1, lat1, lon2, lat2):
    p = np.pi / 180
    a = (np.sin((lat2 - lat1) * p / 2) ** 2
         + np.cos(lat1 * p) * np.cos(lat2 * p) * np.sin((lon2 - lon1) * p / 2) ** 2)
    return 2 * EARTH_R * np.arcsin(np.sqrt(a))


def local_projector(lon0, lat0):
    """Equirectangular projection about (lon0, lat0) -> metres.

    Error is well under 0.5% for trips spanning < ~100 km, which covers every trip
    here (max extent ~80 km). Document/justify this choice (or swap for pyproj UTM).
    """
    k = np.pi / 180 * EARTH_R
    c = np.cos(np.radians(lat0))

    def to_xy(lon, lat):
        return np.column_stack([(np.asarray(lon) - lon0) * k * c,
                                (np.asarray(lat) - lat0) * k])
    return to_xy


def point_segment_distance(P, A, B):
    """Distance from points P (n,2) to segments A->B (n,2) and fraction t along segment."""
    AB = B - A
    L2 = (AB ** 2).sum(1)
    L2 = np.where(L2 == 0, 1e-12, L2)
    t = np.clip(((P - A) * AB).sum(1) / L2, 0, 1)
    C = A + AB * t[:, None]
    return np.sqrt(((P - C) ** 2).sum(1)), t
