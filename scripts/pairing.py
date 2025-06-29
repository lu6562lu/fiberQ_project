import pandas as pd
from obspy import read, Stream
from scripts.signal_processing import integrate_stream

def generate_node_pairs(start_node_idx, end_node_idx, window_length, step):
    """
    Generate sequential node index pairs using a sliding window approach.

    Parameters:
        start_node_idx (int): Index of the starting node.
        end_node_idx (int): Index of the ending node (exclusive).
        window_length (int): Number of nodes in each window.
        step (int): Step size to slide the window.

    Returns:
        List[Tuple[int, int]]: List of node index pairs (start, end).
    """
    meter_to_node_df = pd.read_csv("meter_to_node_map.csv")
    node_list = meter_to_node_df["Node"].tolist()
    pairs = []
    for i in range(start_node_idx, end_node_idx - window_length + 1, step):
        start_node = node_list[i]
        end_node = node_list[i + window_length]
        pairs.append((start_node, end_node))
    return pairs

def generate_node_pairs_with_depth(start_node_idx, end_node_idx, window_length, step, mapping_file):
    """
    Generate node and meter index pairs using a sliding window, with node-to-meter mapping.

    Parameters:
        start_node_idx (int): Starting index in the node list.
        end_node_idx (int): Ending index in the node list (exclusive).
        window_length (int): Window length (number of nodes).
        step (int): Step size for sliding window.
        mapping_file (str): Path to CSV file with 'Node' and 'Meter' columns.

    Returns:
        List[Dict]: List of dictionaries containing 'Node' and 'Meter' pairs.
    """
    meter_to_node_df = pd.read_csv(mapping_file)
    node_list = meter_to_node_df["Node"].tolist()
    meter_list = meter_to_node_df["Meter"].tolist()

    pairs = []
    for i in range(start_node_idx, end_node_idx - window_length + 1, step):
        pairs.append({
            "Node": (node_list[i], node_list[i + window_length]),
            "Meter": (meter_list[i], meter_list[i + window_length])
        })
    return pairs

def process_and_save_data_for_pairs(node_pairs, fiber_data_path, start_time, end_time, waveform_type):
    """
    Process fiber strainrate data between node pairs using waveform integration,
    and return a dictionary of processed waveform Streams.

    Parameters:
        node_pairs (List[Tuple[int, int]]): List of node index pairs.
        fiber_data_path (str): Path to miniSEED or SAC file.
        start_time (UTCDateTime): Start time for reading data.
        end_time (UTCDateTime): End time for reading data.
        waveform_type (str): 'acc', 'vel', or 'dis' (determines how to integrate waveforms).

    Returns:
        Dict[Tuple[int, int], Stream]: Dictionary with node pair keys and processed Stream values.
    """
    fiber_data = read(fiber_data_path, starttime=start_time, endtime=end_time)
    processed_data = {}

    for start_node, end_node in node_pairs:
        print(f"Processing data for nodes {start_node} and {end_node}...")
        try:
            tr_u = fiber_data[start_node]
            tr_l = fiber_data[end_node]
        except IndexError:
            print(f"Error: Node indices {start_node} or {end_node} not found in data.")
            continue

        st_strainrate = Stream(traces=[tr_u, tr_l])
        st_processed = integrate_stream(st_strainrate, waveform_type)
        processed_data[(start_node, end_node)] = st_processed
        print(f"Data for nodes {start_node} and {end_node} processed and saved.")

    return processed_data
