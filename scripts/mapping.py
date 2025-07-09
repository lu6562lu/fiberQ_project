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
        # 計算誤差欄位
        df_map["error"] = (df_map["depth"] - depth).abs()

        # 找最近的 fiber（無論是否有 borehole）
        fiber_row = df_map.loc[df_map["error"].idxmin()]
        fiber_info = str(int(fiber_row["fiber"]))
        fiber_error = fiber_row["error"]
        fiber_depth = fiber_row["depth"]

        # 找 10 公尺內最近的有 borehole 的行
        borehole_candidates = df_map[(df_map["error"] <= 10) & (df_map["borehole"].notna())]
        if not borehole_candidates.empty:
            borehole_row = borehole_candidates.loc[borehole_candidates["error"].idxmin()]
            borehole_info = borehole_row["borehole"]
            borehole_error = borehole_row["error"]
        else:
            borehole_info = None
            borehole_error = None

        return {
            "depth": fiber_depth,
            "fiber": fiber_info,
            "fiber_error": fiber_error,
            "borehole": borehole_info,
            "borehole_error": borehole_error
        }

    info1 = find_nearest(d1)
    info2 = find_nearest(d2)

    print(f"[FIBER] {d1}m → #{info1['fiber']} (misfit: {info1['fiber_error']:.1f}m)")
    print(f"[FIBER] {d2}m → #{info2['fiber']} (misfit: {info2['fiber_error']:.1f}m)")
    print(f"[BOREHOLE] {d1}m → {info1['borehole']} (misfit: {info1['borehole_error']:.1f}m)" if info1['borehole_error'] is not None else f"[BOREHOLE] {d1}m → None")
    print(f"[BOREHOLE] {d2}m → {info2['borehole']} (misfit: {info2['borehole_error']:.1f}m)" if info2['borehole_error'] is not None else f"[BOREHOLE] {d2}m → None")

    return {
        "fiber": (info1["fiber"], info2["fiber"]),
        "borehole": (info1["borehole"], info2["borehole"])
    }