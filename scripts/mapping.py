import pandas as pd

def load_velocity_data(csv_path):
    return pd.read_csv(csv_path)

def average_velocity(depth_pair, velocity_data):
    start_depth, end_depth = depth_pair
    depth_range_data = velocity_data[
        (velocity_data['depth'] >= start_depth) & (velocity_data['depth'] <= end_depth)
    ]
    return depth_range_data['velocity'].mean()

def load_depth_mapping(csv_path):
    return pd.read_csv(csv_path)

def define_depth_combinations(depth_pair, df_map):
    d1, d2 = depth_pair

    def find_nearest(depth):
        idx = (df_map["depth"] - depth).abs().idxmin()
        row = df_map.loc[idx]
        return {
            "depth": row["depth"],
            "fiber": str(int(row["fiber"])),
            "borehole": row["borehole"] if pd.notna(row["borehole"]) else None,
            "error": abs(row["depth"] - depth)
        }

    info1 = find_nearest(d1)
    info2 = find_nearest(d2)

    print(f"[FIBER] {d1}m → #{info1['fiber']} (misfit: {info1['error']:.1f}m)")
    print(f"[FIBER] {d2}m → #{info2['fiber']} (misfit: {info2['error']:.1f}m)")
    print(f"[BOREHOLE] {d1}m → {info1['borehole']} (misfit: {info1['error']:.1f}m)")
    print(f"[BOREHOLE] {d2}m → {info2['borehole']} (misfit: {info2['error']:.1f}m)")

    return {
        "fiber": (info1["fiber"], info2["fiber"]),
        "borehole": (info1["borehole"], info2["borehole"])
    }
