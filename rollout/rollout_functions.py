# === rollout_functions.py ===
import json
import random

import pandas as pd
import torch
import numpy as np
import math

import matplotlib.pyplot as plt
import pickle

from tqdm import tqdm


# Helper functions
def get_empty_distance_and_time(start_zone, dest_zone, matrix, default_distance=10.0, default_time=30.0):
    """
    Return empty travel distance (miles) and duration (minutes) between two zones.
    Handle NaN entries by assigning default large values.
    """
    try:
        distance = matrix[start_zone, dest_zone, 2]  # mean trip distance
        duration = matrix[start_zone, dest_zone, 0]  # mean trip duration
    except IndexError:
        distance = np.nan
        duration = np.nan

    if np.isnan(distance):
        distance = default_distance
    if np.isnan(duration):
        duration = default_time

    return distance, duration


def advance_time_bin(current_bin, minutes):
    """
    Advance time by given minutes, and wrap around 96 bins (24h).
    """
    bins_advanced = math.ceil(minutes / 15)
    return (current_bin + bins_advanced) % 96, bins_advanced


def stay_strategy(current_zone):
    """
    Always stay in the current zone.
    """
    return current_zone


# Main rollout function
def rollout_one(start_zone, start_time_bin, policy, q_net=None, zone_id_map=None,
                neighbor_dict=None, trip_df=None, matrix=None, device='cpu', valid_sa_pairs = None):
    """
    Perform one rollout simulation starting from a given zone and time bin, under a specified policy.
    """
    s_zone = start_zone
    t_bin = start_time_bin
    total_income = 0.0
    bins_elapsed = 0
    max_bins = 32
    stay_no_trip_counter = 0

    while bins_elapsed < max_bins:
        # Select action based on policy
        if policy == 'q_learning':
            # # Filter global zone set by whether (s_zone, a_zone) has been seen
            # candidates = [a_zone for a_zone in zone_id_map.keys() if (s_zone, a_zone) in valid_sa_pairs]
            # if not candidates:
            #     a_zone = s_zone
            # else:
            #     s_tensor = torch.tensor([zone_id_map[s_zone]] * len(candidates), dtype=torch.long).to(device)
            #     t_tensor = torch.tensor([t_bin] * len(candidates), dtype=torch.long).to(device)
            #     a_tensor = torch.tensor([zone_id_map[a] for a in candidates], dtype=torch.long).to(device)
            #     with torch.no_grad():
            #         q_values = q_net(s_tensor, t_tensor, a_tensor)
            #         # print(q_values)
            #     best_idx = torch.argmax(q_values).item()
            #     a_zone = candidates[best_idx]
            candidates = neighbor_dict.get(s_zone, [s_zone])
            candidates = [a for a in candidates if a in zone_id_map]
            if not candidates:
                candidates = [s_zone]
            s_tensor = torch.tensor([zone_id_map[s_zone]] * len(candidates), dtype=torch.long).to(device)
            t_tensor = torch.tensor([t_bin] * len(candidates), dtype=torch.long).to(device)
            a_tensor = torch.tensor([zone_id_map[a] for a in candidates], dtype=torch.long).to(device)
            with torch.no_grad():
                q_values = q_net(s_tensor, t_tensor, a_tensor)
            best_idx = torch.argmax(q_values).item()
            a_zone = candidates[best_idx]
        elif policy == 'random':
            a_zone = random.choice(trip_df['PULocationID'].unique())
        elif policy == 'stay':
            a_zone = stay_strategy(s_zone)
        else:
            raise ValueError(f"Unsupported policy: {policy}")

        # Apply empty travel cost if move
        if a_zone != s_zone:
            empty_distance, empty_duration = get_empty_distance_and_time(s_zone, a_zone, matrix)
            empty_cost = empty_distance * 0.25  # USD per mile
            total_income -= empty_cost
            t_bin, bins_used = advance_time_bin(t_bin, empty_duration)
            bins_elapsed += bins_used
            if bins_elapsed >= max_bins:
                break
            s_zone = a_zone
        else:
            empty_cost = 0.0

        # After move (or stay), try to find a trip
        candidate_trips = trip_df[
            (trip_df['PULocationID'] == s_zone) &
            (trip_df['pickup_time_bin'] == t_bin)
        ]

        if not candidate_trips.empty:
            trip = candidate_trips.sample(1).iloc[0]
            fare = float(trip['FareAmount'])
            trip_duration = float(trip['TripDuration'])
            dropoff_zone = int(trip['DOLocationID'])

            total_income += fare
            t_bin, bins_used = advance_time_bin(t_bin, trip_duration)
            bins_elapsed += bins_used
            s_zone = dropoff_zone
        else:
            t_bin = (t_bin + 1) % 96
            bins_elapsed += 1
            stay_no_trip_counter += 1
            if stay_no_trip_counter >= 2 and policy == 'q_learning':
                candidates = neighbor_dict.get(s_zone, [s_zone])
                candidates = [zone for zone in candidates if (zone != s_zone) and (zone in zone_id_map)]
                if candidates:
                    a_zone = random.choice(candidates)
                    empty_distance, empty_duration = get_empty_distance_and_time(s_zone, a_zone, matrix)
                    empty_cost = empty_distance * 0.5
                    total_income -= empty_cost
                    t_bin, bins_used = advance_time_bin(t_bin, empty_duration)
                    bins_elapsed += bins_used
                    s_zone = a_zone
                stay_no_trip_counter = 0  # reset counter after forced move

    return total_income

def rollout_one_debug(start_zone, start_time_bin, policy, q_net=None, zone_id_map=None,
                neighbor_dict=None, trip_df=None, matrix=None, device='cpu'):
    """
    Debugging version of rollout_one with detailed logging information.
    """
    s_zone = start_zone
    t_bin = start_time_bin
    total_income = 0.0
    bins_elapsed = 0
    max_bins = 32
    num_trips_taken = 0
    total_empty_cost = 0.0
    total_empty_time = 0.0
    total_moves = 0
    total_stays = 0
    stay_no_trip_counter = 0

    print(f"\n=== Start rollout from Zone {s_zone}, TimeBin {t_bin} ({policy}) ===")

    while bins_elapsed < max_bins:
        if policy == 'q_learning':
            candidates = neighbor_dict.get(s_zone, [s_zone])
            candidates = [a for a in candidates if a in zone_id_map]
            if not candidates:
                candidates = [s_zone]
            s_tensor = torch.tensor([zone_id_map[s_zone]] * len(candidates), dtype=torch.long).to(device)
            t_tensor = torch.tensor([t_bin] * len(candidates), dtype=torch.long).to(device)
            a_tensor = torch.tensor([zone_id_map[a] for a in candidates], dtype=torch.long).to(device)
            with torch.no_grad():
                q_values = q_net(s_tensor, t_tensor, a_tensor)
            best_idx = torch.argmax(q_values).item()
            a_zone = candidates[best_idx]
        elif policy == 'random':
            candidates = neighbor_dict.get(s_zone, [s_zone])
            candidates = [a for a in candidates if a in zone_id_map]

            a_zone = random.choice(candidates)
        elif policy == 'stay':
            a_zone = s_zone
        else:
            raise ValueError(f"Unsupported policy: {policy}")

        # Move or Stay decision
        if a_zone != s_zone:
            empty_distance, empty_duration = get_empty_distance_and_time(s_zone, a_zone, matrix)
            empty_cost = empty_distance * 0.5  # USD per mile
            total_income -= empty_cost
            total_empty_cost += empty_cost
            total_empty_time += empty_duration
            t_bin, bins_used = advance_time_bin(t_bin, empty_duration)
            bins_elapsed += bins_used
            total_moves += 1
            print(f"🚗 Move from Zone {s_zone} to Zone {a_zone} | EmptyCost = ${empty_cost:.2f} | EmptyTime = {empty_duration:.1f} mins")
            if bins_elapsed >= max_bins:
                break
            s_zone = a_zone
        else:
            empty_cost = 0.0
            total_stays += 1
            stay_no_trip_counter += 1
            if stay_no_trip_counter >= 2 and policy == 'q_learning':
                candidates = neighbor_dict.get(s_zone, [s_zone])
                candidates = [zone for zone in candidates if (zone != s_zone) and (zone in zone_id_map)]
                if candidates:
                    a_zone = random.choice(candidates)
                    empty_distance, empty_duration = get_empty_distance_and_time(s_zone, a_zone, matrix)
                    empty_cost = empty_distance * 0.5
                    total_income -= empty_cost
                    t_bin, bins_used = advance_time_bin(t_bin, empty_duration)
                    bins_elapsed += bins_used
                    s_zone = a_zone
                stay_no_trip_counter = 0  # reset counter after forced move
                print(f"🛑 Force move to {s_zone} |  EmptyCost = ${empty_cost:.2f} | EmptyTime = {empty_duration:.1f} mins.")
            else:
                print(f"🛑 Stay at Zone {s_zone} | No empty cost.")
        # Try to find a trip
        candidate_trips = trip_df[
            (trip_df['PULocationID'] == s_zone) &
            (trip_df['pickup_time_bin'] == t_bin)
        ]

        if not candidate_trips.empty:
            trip = candidate_trips.sample(1).iloc[0]
            fare = float(trip['FareAmount'])
            trip_duration = float(trip['TripDuration'])
            dropoff_zone = int(trip['DOLocationID'])

            total_income += fare
            num_trips_taken += 1
            t_bin, bins_used = advance_time_bin(t_bin, trip_duration)
            bins_elapsed += bins_used
            print(f"✅ Picked up a trip: Fare = ${fare:.2f}, TripTime = {trip_duration:.1f} mins, Dropoff at Zone {dropoff_zone}")

            s_zone = dropoff_zone
        else:
            t_bin = (t_bin + 1) % 96
            bins_elapsed += 1
            print(f"❌ No trip available at Zone {s_zone}, waiting 15 minutes...")

    # Final Summary
    print(f"\n=== End of rollout ===")
    print(f"Total income: ${total_income:.2f}")
    print(f"Total trips taken: {num_trips_taken}")
    print(f"Total empty cost: ${total_empty_cost:.2f}")
    print(f"Total empty time: {total_empty_time:.1f} minutes")
    print(f"Total moves: {total_moves}")
    print(f"Total stays: {total_stays}")

    return total_income


# Batch rollout
def batch_rollout(num_samples, q_net, zone_id_map, neighbor_dict, trip_df, matrix, device='cpu',
                  policies=('q_learning', 'random', 'stay'), random_choose=False):
    results = {policy: [] for policy in policies}
    if not random_choose:

        start_zones = list(zone_id_map.keys())
        start_time_bin = 40  # 10am
        for start_zone in tqdm(start_zones, desc="Rollout for each zone"):
            for policy in policies:
                income = rollout_one(
                    start_zone=start_zone,
                    start_time_bin=start_time_bin,
                    policy=policy,
                    q_net=q_net,
                    zone_id_map=zone_id_map,
                    neighbor_dict=neighbor_dict,
                    trip_df=trip_df,
                    matrix=matrix,
                    device=device,
                )
                results[policy].append(income)
    else:
        available_zones = trip_df['PULocationID'].unique()
        # high_demand_zones = [
        #     132, 138, 161, 237, 230
        # ]
        # available_time_bins = trip_df['pickup_time_bin'].unique()
        for _ in range(num_samples):
            start_zone = int(random.choice(available_zones))
            start_time_bin = 40

            for policy in policies:
                income = rollout_one(start_zone, start_time_bin, policy,
                                     q_net=q_net,
                                     zone_id_map=zone_id_map,
                                     neighbor_dict=neighbor_dict,
                                     trip_df=trip_df,
                                     matrix=matrix,
                                     device=device,
                                     )
                results[policy].append(income)

    return results


# Summarize and print results
def summarize_results(results_dict):
    summary = {}
    for policy, incomes in results_dict.items():
        mean_income = np.mean(incomes)
        std_income = np.std(incomes)
        summary[policy] = (mean_income, std_income)
    return summary


def print_summary(summary):
    print("\n=== Rollout Simulation Results ===")
    for policy, (mean_income, std_income) in summary.items():
        print(f"{policy}: Mean = ${mean_income:.2f}, Std = ${std_income:.2f}")


def generate_neighbor_dict_from_matrix(matrix_path, output_path='neighbor_dict.json', topk=10):
    """
    Generate neighbor dictionary based on mean trip distance from intra_zone_matrix.
    Each zone selects topk nearest neighbors (smallest distance) + itself.
    """
    print(f"Loading intra-zone matrix from '{matrix_path}'...")
    matrix = np.load(matrix_path)  # shape (266, 266, 4)
    distance_matrix = matrix[:, :, 2]  # Use mean trip distance (index 2)

    neighbor_dict = {}

    num_zones = distance_matrix.shape[0]
    for zone in range(num_zones):
        distances = distance_matrix[zone]

        # Handle NaNs: temporarily set nan to very large distance
        distances = np.where(np.isnan(distances), np.inf, distances)

        # Exclude self temporarily to find topk nearest others
        distances_without_self = distances.copy()
        distances_without_self[zone] = np.inf

        nearest_indices = np.argsort(distances_without_self)[:topk]

        # Add self zone back
        neighbors = list(nearest_indices)
        neighbors.append(zone)

        # Force all IDs to pure Python int (important!)
        neighbors = sorted([int(n) for n in neighbors])

        neighbor_dict[int(zone)] = neighbors  # Make sure keys are also int

    # Save as JSON
    with open(output_path, 'w') as f:
        json.dump(neighbor_dict, f)

    print(f"✅ neighbor_dict saved to '{output_path}' with {len(neighbor_dict)} zones.")


def plot_results():
    with open('rollout_results.pkl', 'rb') as f:
        results = pickle.load(f)

    # Extract incomes
    q_incomes = results['q_learning']
    random_incomes = results['random']
    stay_incomes = results['stay']

    # === Create Boxplot ===
    plt.figure(figsize=(8, 6))

    plt.boxplot(
        [stay_incomes, q_incomes, random_incomes],
        labels=['Stay', 'Q-learning', 'Random'],
        showmeans=True
    )

    plt.ylabel('Total Income over 8 Hours (USD)')
    plt.title('Boxplot of Driver Total Income for Different Strategies')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()

    df = pd.DataFrame(results)
    df['zone_id'] = df.index

    plt.figure(figsize=(12, 6))
    plt.plot(df['zone_id'], df['q_learning'], label='Q-learning', marker='o', alpha=0.7)
    plt.plot(df['zone_id'], df['random'], label='Random', marker='s', alpha=0.7)
    plt.plot(df['zone_id'], df['stay'], label='Stay', marker='^', alpha=0.7)

    plt.xlabel('Zone ID (Start Zone)')
    plt.ylabel('Total Income over 8 Hours (USD)')
    plt.title('Rollout Income per Zone for Each Strategy')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

