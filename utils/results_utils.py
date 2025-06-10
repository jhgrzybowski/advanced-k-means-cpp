import os
import json
import random
import numpy as np
from utils.data_utils import load_gml_to_delay_graph
from utils.experiment_utils import compute_latencies_for_experiment

results_dir = "results/os3e"

def save_results_to_json(
    gml_file,
    clustering_fns,
    propagation_speed_km_per_ms,
    kmax,
    enhanced_runs,
    seed,
    enhanced_k_means_kwargs=None
):
    """
    Args:
        gml_file (str): Path to network topology in GML format.
        clustering_fns (dict): Callable algorithm functions {algorithm_name: clustering_fn}
        propagation_speed_km_per_ms (float): Signal propagation speed in km/ms.
        kmax (int): Max number of controllers to test.
        enhanced_runs (int): Number of stochastic runs for each k for enhanced k-means.
        seed (int or None): Seed for reproducibility.
        enhanced_k_means_kwargs (dict): Weight arguments passed to enhanced_k_means_fn.

    Saves:
        Results to 'results/' directory in enhanced_kmeans_results.json format.
    """
    os.makedirs(results_dir, exist_ok=True)
    G = load_gml_to_delay_graph(gml_file, propagation_speed_km_per_ms)
    rng = random.Random(seed)
    kwargs = enhanced_k_means_kwargs or {}

    enhanced_results = {
        "runs": enhanced_runs,
        "k_range": list(range(1, kmax + 1)),
        "data": []
    }

    for k in range(1, kmax + 1):
        avg_delays = []
        centers_per_run = []
        clusters_per_run = []

        for run in range(enhanced_runs):
            controllers, clusters = clustering_fns["enhanced_k_means"](G, k, rng, **kwargs)
            centers_per_run.append(list(controllers))
            # Ensure clusters are serializable as {str: list}
            clusters_serializable = {str(int(c)): list(map(int, members)) for c, members in clusters.items()}
            clusters_per_run.append(clusters_serializable)

            avg_delay, _ = compute_latencies_for_experiment(G, k, controllers, clusters)
            avg_delays.append(float(avg_delay[0]))  # avg_delay is [value], we want value

        # Statistics
        mean = float(np.mean(avg_delays))
        std = float(np.std(avg_delays))
        max_v = float(np.max(avg_delays))
        min_v = float(np.min(avg_delays))

        enhanced_results["data"].append({
            "k": k,
            "avg_delays": avg_delays,
            "mean": mean,
            "std": std,
            "max": max_v,
            "min": min_v,
            "centers": centers_per_run,
            "clusters": clusters_per_run,
        })

    with open(f"{results_dir}/enhanced_k-means_results.json", "w") as f:
        json.dump(enhanced_results, f, indent=2)
    print("Results successfully saved to results/enhanced_k-means_results.json")

    advanced_results = {
        "k_range": list(range(1, kmax + 1)),
        "data": []
    }

    # Final delays list after experiments of Advanced K-Means
    avg_delays_advanced = []
    max_delays_advanced = []

    for k_idx, k in enumerate(range(1, kmax + 1)):

        # --- Advanced K-Means latency measurements ---
        controllers, clusters = clustering_fns["advanced_k_means"](G, k)

        advanced_avg, advanced_max = compute_latencies_for_experiment(G, k, controllers, clusters)

        # Experiments result lists for Advanced K-Means
        avg_delays_advanced.append(np.mean(advanced_avg))
        max_delays_advanced.append(np.max(advanced_max))

    for k_idx, k in enumerate(range(1, kmax + 1)):
        avg_delay = float(avg_delays_advanced[k_idx])
        advanced_results["data"].append({
            "k": k,
            "mean": avg_delay
        })

    with open(f"{results_dir}/advanced_k-means_results.json", "w") as f:
        json.dump(advanced_results, f, indent=2)
    print("Results successfully saved to results/advanced_k-means_results.json")