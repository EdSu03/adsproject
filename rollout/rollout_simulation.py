# === rollout_simulation.py ===
import random

import pandas as pd
import numpy as np
import torch
import json
from rollout_functions import batch_rollout, summarize_results, print_summary, rollout_one_debug, \
    generate_neighbor_dict_from_matrix, plot_results
from q_network import QNetwork
import pickle


def get_time_bin(timestamp):
    return timestamp.hour * 4 + timestamp.minute // 15


def main():
    # === Step 1: Paths ===
    trip_data_path = '../cleaned_data/cleaned_yellow_tripdata_2024-03.parquet'
    q_network_model_path = '../Q_learning/q_network_trip2trip_20_epochs.pt'
    intra_zone_matrix_path = '../intra_zone_matrix.npy'
    neighbor_dict_path = '../neighbor_dict_10.json'

    # === Step 2: Load trip data ===
    trip_df = pd.read_parquet(trip_data_path)

    # Preprocess: create pickup_time_bin
    trip_df['PickupDatetime'] = pd.to_datetime(trip_df['PickupDatetime'])
    trip_df['pickup_time_bin'] = trip_df['PickupDatetime'].apply(get_time_bin)

    print("✅ Trip dataframe loaded and processed.")

    # === Step 3: Load neighbor_dict ===
    with open(neighbor_dict_path, 'r') as f:
        neighbor_dict = json.load(f)
    neighbor_dict = {int(k): v for k, v in neighbor_dict.items()}  # Ensure keys are int
    print("✅ Neighbor dictionary loaded.")

    # === Step 4: Load intra-zone distance matrix ===
    intra_zone_matrix = np.load(intra_zone_matrix_path)
    print("✅ Intra-zone distance matrix loaded.")

    # === Step 5: Prepare zone_id_map and inv_zone_map ===
    samples_path = '../data_for_q_learning/q_learning_trip_to_trip_penalty_2024-01.csv'
    df = pd.read_csv(samples_path)

    # Encode all zone IDs into continuous integer IDs
    all_zone_ids = pd.unique(df[['s_zone', 'a_zone', 's_prime_zone']].values.ravel())
    zone_id_map = {z: i for i, z in enumerate(sorted(all_zone_ids))}
    # zone_ids = pd.unique(trip_df[['PULocationID', 'DOLocationID']].values.ravel())
    # zone_id_map = {zone_id: idx for idx, zone_id in enumerate(sorted(zone_ids))}
    # inv_zone_map = {idx: zone_id for zone_id, idx in zone_id_map.items()}
    num_zones = len(zone_id_map)
    num_time_bins = 96  # 15-min time bins

    print(f"✅ Zone ID mapping created: {num_zones} zones.")

    # === Step 6: Load trained Q-network ===
    q_net = QNetwork(num_zones, num_time_bins)
    q_net.load_state_dict(torch.load(q_network_model_path))
    q_net.eval()
    device = 'cpu'  # or 'cuda' if you have GPU
    print("✅ Q-network model loaded.")

    # === Step 7: Run batch rollout === #

    results = batch_rollout(
        num_samples=500,
        q_net=q_net,
        zone_id_map=zone_id_map,
        neighbor_dict=neighbor_dict,
        trip_df=trip_df,
        matrix=intra_zone_matrix,
        device=device,
        random_choose=True
    )

    # === Step 8: Summarize and print results === #
    summary = summarize_results(results)
    print_summary(summary)

    # === Step 9: Save results ===
    output_path = 'rollout_results.pkl'
    with open(output_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"✅ Rollout results saved to '{output_path}'.")

    # === Debugging single rollout === #
    # high_demand_zones = [
    #     # 43, 48, 50, 68, 74, 75, 87, 88, 90, 100, 161, 163, 164, 166, 170, 186, 230, 234, 237
    #     221
    # ]
    # available_zones = trip_df['PULocationID'].unique()
    # start_zone = random.choice(available_zones)
    # start_time_bin = random.choice(trip_df['pickup_time_bin'].unique())
    #
    # # q_data = pd.read_csv('../data_for_q_learning/q_learning_samples_realistic_2024-01_02.csv')
    # #
    # # valid_sa_pairs = set(zip(q_data['s_zone'], q_data['a_zone']))
    # # print(f"✅ Loaded {len(valid_sa_pairs)} valid (s, a) pairs from training data.")
    #
    # print(f"\n=== Debugging rollout starting at Zone {start_zone}, TimeBin {start_time_bin} ===")
    #
    # _ = rollout_one_debug(
    #     start_zone=start_zone,
    #     start_time_bin=start_time_bin,
    #     policy='q_learning',  # ‘q_learning’, 'random', 'stay'
    #     q_net=q_net,
    #     zone_id_map=zone_id_map,
    #     neighbor_dict=neighbor_dict,
    #     trip_df=trip_df,
    #     matrix=intra_zone_matrix,
    #     device='cpu',
    # )


if __name__ == "__main__":
    main()
    plot_results()

