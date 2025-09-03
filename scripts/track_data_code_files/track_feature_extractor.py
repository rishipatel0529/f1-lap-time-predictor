"""
scripts/track_data_code_files/track_feature_extractor.py

Extracts per-corner geometric and speed features for F1 tracks into a flat CSV.
Reads track centerlines (GeoJSON), per-corner apex coordinates, and optional elevation profiles.
Computes radii, inter-corner distances, pit-lane length, elevation deltas, and typical corner speeds.
Intended to generate modeling features like corner_{i}_{radius|distance|elev_change|entry/exit/max_speed}.
"""

import glob
import json
import os
from math import atan2, cos, radians, sin, sqrt

import fastf1 as ff1
import numpy as np
import pandas as pd
from shapely.geometry import LineString

TRACK_META_CSV = "data/track_data_csv_files/track_corners_coords.csv"
CENTERLINE_DIR = "data/track_data_csv_files/track_centerlines_gps_cords"
ELEVATION_DIR = "data/track_data_csv_files/elevation_profiles"
FASTF1_CACHE_DIR = "data/track_data_csv_files/ff1_cache"
OUTPUT_CSV = "data/track_data_csv_files/track_features.csv"
CORNER_BUF_M = 15  # how far upstream/downstream of apex to sample

os.makedirs(FASTF1_CACHE_DIR, exist_ok=True)
ff1.Cache.enable_cache(FASTF1_CACHE_DIR)


def haversine_m(a, b):
    # Great-circle distance in meters between two (lat, lon) points
    R = 6371000.0
    lat1, lon1 = map(radians, a)
    lat2, lon2 = map(radians, b)
    dlat, dlon = lat2 - lat1, lon2 - lon1
    x = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    return 2 * R * atan2(sqrt(x), sqrt(1 - x))


def latlon_to_xy(lat, lon, lat0, lon0):
    # Local tangent-plane approximation turning (lat, lon) into meters relative to an origin (lat0, lon0)
    R = 6371000.0
    x = R * radians(lon - lon0) * cos(radians(lat0))
    y = R * radians(lat - lat0)
    return np.array([x, y])


def fit_circle_radius(p1, p2, p3):
    # Estimate the radius of curvature passing through three points using circumcircle algebra
    A = np.linalg.det([[p1[0], p1[1], 1], [p2[0], p2[1], 1], [p3[0], p3[1], 1]])
    if abs(A) < 1e-6:
        return np.nan

    def sq(p):
        return p[0] ** 2 + p[1] ** 2

    B = np.linalg.det([[sq(p1), p1[1], 1], [sq(p2), p2[1], 1], [sq(p3), p3[1], 1]])
    C = np.linalg.det([[sq(p1), p1[0], 1], [sq(p2), p2[0], 1], [sq(p3), p3[0], 1]])
    ux = B / (2 * A)
    uy = -C / (2 * A)
    return float(np.hypot(ux - p1[0], uy - p1[1]))


def load_centerline_xy(host):
    # Load the track centerline for a given host slug and derive cumulative distance (meters) along it
    gj = json.load(open(f"{CENTERLINE_DIR}/{host}.geojson"))
    coords = gj["features"][0]["geometry"]["coordinates"]
    lon0, lat0 = coords[0]
    xy = np.array([latlon_to_xy(lat, lon, lat0, lon0) for lon, lat in coords])
    ds = np.concatenate([[0], np.cumsum(np.linalg.norm(xy[1:] - xy[:-1], axis=1))])
    return xy, ds, lat0, lon0


def get_elevations_at(distances, elevation_csv):
    # Interpolate elevation values from a per-track profile at requested centerline distances
    prof = pd.read_csv(elevation_csv)
    return np.interp(distances, prof["distance_m"], prof["elevation_m"])


def compute_corner_speeds(lap, apex_dist, L_entry=50, L_exit=80):
    # Slice lap telemetry around an apex and summarize entry/exit/max speeds for that segment
    tel = lap.get_telemetry().add_distance()
    d_start = max(0, apex_dist - L_entry)
    d_end = min(tel["Distance"].max(), apex_dist + L_exit)
    seg = tel[(tel["Distance"] >= d_start) & (tel["Distance"] <= d_end)]
    if seg.empty:
        return (np.nan, np.nan, np.nan)
    speeds = seg["Speed"]
    return (float(speeds.iat[0]), float(speeds.iat[-1]), float(speeds.max()))


def main():
    df = pd.read_csv(TRACK_META_CSV)
    df["year_start"] = df["years"].astype(str).str.split("-", n=1).str[0].astype(int)
    # Restrict to a specific host and period to keep the example deterministic and manageable
    df = df[
        (df["host"] == "russian_grand_prix") & (df["year_start"].between(2019, 2021))
    ].reset_index(drop=True)
    print("Running on:", df[["host", "years", "year_start"]].values.tolist())

    out = []
    for _, r in df.iterrows():
        host = r["host"]

        # Parse pit entry/exit GPS coordinates for pit-lane length calculation
        pit_ent = tuple(map(float, r["pit_entry_coords"].split(",")))
        pit_ext = tuple(map(float, r["pit_exit_coords"].split(",")))

        # Collect available apex coordinates; rows may omit some corner entries
        nc = int(r["corners"])
        apex_ll = []
        for i in range(1, nc + 1):
            val = r.get(f"corner_{i}_coords")
            if pd.isna(val):
                continue
            lat, lon = map(float, str(val).split(","))
            apex_ll.append((lat, lon))

        # Load centerline and construct a cumulative distance axis for projection
        xy, ds, lat0, lon0 = load_centerline_xy(host)

        # Project each apex to the closest centerline point and sort by lap distance
        apex_xy = np.array([latlon_to_xy(lat, lon, lat0, lon0) for lat, lon in apex_ll])
        apex_d = np.array(
            [ds[np.argmin(np.linalg.norm(xy - p, axis=1))] for p in apex_xy]
        )
        order = np.argsort(apex_d)
        apex_d = apex_d[order]

        # Estimate a local curvature radius at each apex using points ± CORNER_BUF_M around it
        line = LineString(xy)
        radii = []
        for d in apex_d:
            p0 = np.array(line.interpolate(max(0, d - CORNER_BUF_M)).coords[0])
            p = np.array(line.interpolate(d).coords[0])
            p2 = np.array(line.interpolate(min(ds[-1], d + CORNER_BUF_M)).coords[0])
            radii.append(fit_circle_radius(p0, p, p2))

        # Compute distances between successive apexes and to the start/finish
        corner_distances = np.diff(np.concatenate([[0], apex_d, [ds[-1]]])).tolist()

        # Pit-lane length derived from geodesic distance between entry and exit markers
        pit_len = haversine_m(pit_ent, pit_ext)

        # Interpolate elevation at apex distances and form deltas between corners
        pat = os.path.join(ELEVATION_DIR, f"{host}*elevation_profile.csv")
        files = glob.glob(pat)
        if not files:
            raise FileNotFoundError(f"No elevation profile for {host}")
        elevs = get_elevations_at(apex_d, files[0])
        elev_changes = np.diff(elevs).tolist()

        # Pull a representative lap (fastest) to summarize entry/exit/max speeds around each apex
        year = int(str(r["years"]).split("-")[0])
        sess = ff1.get_session(year, host, "Race")
        sess.load()
        lap = sess.laps.pick_fastest()
        spds = [compute_corner_speeds(lap, d, L_entry=30, L_exit=30) for d in apex_d]
        ent_speeds, exit_speeds, max_speeds = zip(*spds)

        out.append(
            {
                "host": host,
                "corner_radii": radii,
                "corner_distances": corner_distances,
                "pit_lane_length": float(pit_len),
                "elevation_changes": elev_changes,
                "corner_entry_speeds": list(ent_speeds),
                "corner_exit_speeds": list(exit_speeds),
                "corner_max_speeds": list(max_speeds),
            }
        )

    # Expand list-valued fields into per-corner columns so downstream modeling can select by prefix
    df_feats = pd.DataFrame(out)
    max_n = df_feats["corner_radii"].apply(len).max()
    for i in range(max_n):
        df_feats[f"corner_{i+1}_radius"] = df_feats["corner_radii"].apply(
            lambda x: x[i] if i < len(x) else np.nan
        )
        df_feats[f"corner_{i+1}_distance"] = df_feats["corner_distances"].apply(
            lambda x: x[i] if i < len(x) else np.nan
        )
        df_feats[f"corner_{i+1}_elev_change"] = df_feats["elevation_changes"].apply(
            lambda x: x[i] if i < len(x) else np.nan
        )
        df_feats[f"corner_{i+1}_entry_speed"] = df_feats["corner_entry_speeds"].apply(
            lambda x: x[i] if i < len(x) else np.nan
        )
        df_feats[f"corner_{i+1}_exit_speed"] = df_feats["corner_exit_speeds"].apply(
            lambda x: x[i] if i < len(x) else np.nan
        )
        df_feats[f"corner_{i+1}_max_speed"] = df_feats["corner_max_speeds"].apply(
            lambda x: x[i] if i < len(x) else np.nan
        )

    # Remove the original list columns now that the wide, per-corner fields exist
    df_feats.drop(
        columns=[
            "corner_radii",
            "corner_distances",
            "elevation_changes",
            "corner_entry_speeds",
            "corner_exit_speeds",
            "corner_max_speeds",
        ],
        inplace=True,
    )

    df_feats.to_csv(OUTPUT_CSV, index=False)
    print(f"Wrote expanded track features to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
