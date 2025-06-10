import os
import random
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Arial'

from utils.data_utils import load_gml_to_delay_graph
from utils.experiment_utils import compute_latencies_for_experiment
from utils.plot_utils import plot_latency_comparison, plot_enhanced_kmeans_experiment

def run_latency_experiment_compare(
    gml_file,
    clustering_fns,
    propagation_speed_km_per_ms,
    kmax,
    seed,
    enhanced_k_means_kwargs=None,
    log_k_values=None
):
    """
    Runs latency experiments for a given topology and multiple clustering algorithms.
    Args:
        gml_file (str): Path to the GML file.
        clustering_fns (dict): Callable algorithm functions {algorithm_name: clustering_fn}
        propagation_speed_km_per_ms (float): Signal propagation speed.
        seed (int): Random seed.
        kmax (int): Maximum number of controllers to test.
        enhanced_k_means_kwargs (dict): Weight arguments passed to advanced_k_means_fn.
        log_k_values (list or int, optional): k values for detailed logging.
    Saves:
        Comparison plots for average and max latency.
    """

    os.makedirs("plots", exist_ok=True)

    G = load_gml_to_delay_graph(gml_file, propagation_speed_km_per_ms=propagation_speed_km_per_ms)
    k_values = list(range(1, kmax + 1))

    if log_k_values is None:
        log_k_values = []
    if isinstance(log_k_values, int):
        log_k_values = [log_k_values]

    rng = random.Random(seed)

    avg_latencies = {}
    max_latencies = {}

    for name, fn in clustering_fns.items():
        avg_times = []
        max_times = []

        print("=" * 80)
        print(f"## Algorithm: {name.upper()} ##")

        for k in k_values:
            # Select kwargs for enhanced k-means
            if name == "enhanced_k_means":
                kwargs = enhanced_k_means_kwargs or {}
                controllers, clusters = fn(G, k, rng, **kwargs)
            else:
                controllers, clusters = fn(G, k)

            # Logging if k is selected
            if k in log_k_values:
                print("#" * 60)
                print(f"k={k}: Controllers (IDs): {controllers}")
                for ctrl in controllers:
                    ctrl_label = G.nodes[ctrl].get('label', str(ctrl))
                    members_labels = [G.nodes[n].get('label', str(n)) for n in clusters[ctrl]]
                    print(f"  Controller {ctrl} ({ctrl_label}): {members_labels}")

            # Compute latencies using helper
            avg_delay_list, max_delay_list = compute_latencies_for_experiment(G, k, controllers, clusters)
            avg_latency = avg_delay_list[0] if isinstance(avg_delay_list, list) else avg_delay_list
            max_latency = max_delay_list[0] if isinstance(max_delay_list, list) else max_delay_list

            avg_times.append(avg_latency)
            max_times.append(max_latency)

            if k in log_k_values:
                print(f"  [k={k}] avg_latency = {avg_latency:.4f}, max_latency = {max_latency:.4f}")

        print("=" * 80)
        avg_latencies[name] = avg_times
        max_latencies[name] = max_times

    plot_latency_comparison(
        k_values,
        avg_latencies,
        max_latencies,
        clustering_fns,
        experiment_name="Advanced K-Means vs Enhanced K-Means++",
        topology_name="Internet2 OS3E"
    )

def run_enhanced_kmeans_experiment(
    gml_file,
    clustering_fns,
    propagation_speed_km_per_ms,
    kmax,
    enhanced_runs,
    seed,
    enhanced_k_means_kwargs=None
):
    """
    Run latency experiments comparing advanced k-means and enhanced (probabilistic seeding) k-means++.
    For enhanced k-means, perform multiple runs per k to gather statistics on the random outcomes.
    Plots average and max delay for both algorithms (mean and stddev for enhanced k-means).

    Args:
        gml_file (str): Path to network topology in GML format.
        clustering_fns (dict): Callable algorithm functions {algorithm_name: clustering_fn}
        propagation_speed_km_per_ms (float): Signal propagation speed in km/ms.
        kmax (int): Max number of controllers to test.
        enhanced_runs (int): Number of stochastic runs for each k for enhanced k-means.
        seed (int or None): Seed for reproducibility.
        enhanced_k_means_kwargs (dict): Weight arguments passed to advanced_k_means_fn.

    Saves:
        Plots to 'plots/' directory.
    """
    import os
    os.makedirs("plots", exist_ok=True)

    # Load topology
    G = load_gml_to_delay_graph(gml_file, propagation_speed_km_per_ms)
    k_values = list(range(1, kmax + 1))

    # Final delays list after experiments of Advanced K-Means
    avg_delays_advanced = []
    max_delays_advanced = []

    # Final average delays and std deviation list after experiments of Enhanced K-Means++ after set number of runs
    avg_delays_enhanced = []
    std_avg_delays_enhanced = []
    max_delays_enhanced = []
    std_max_delays_enhanced = []

    # Initialize rng with given seed
    rng = random.Random(seed)

    # Kwargs could be optional
    kwargs = enhanced_k_means_kwargs or {}

    # Run experiment for k=1 to kmax
    for k in range(1, kmax + 1):

        # --- Advanced K-Means latency measurements ---
        controllers, clusters = clustering_fns["advanced_k_means"](G, k)

        advanced_avg, advanced_max = compute_latencies_for_experiment(G, k, controllers, clusters)

        # Experiments result lists for Advanced K-Means
        avg_delays_advanced.append(np.mean(advanced_avg))
        max_delays_advanced.append(np.max(advanced_max))

        # --- Enhanced K-Means++ latency measurements ---
        enhanced_avg = []
        enhanced_max = []

        for run in range(enhanced_runs):
            controllers_enhanced, clusters_enhanced = clustering_fns["enhanced_k_means"](G, k, rng, **kwargs)

            run_enhanced_avg, run_enhanced_max = compute_latencies_for_experiment(G, k, controllers_enhanced, clusters_enhanced)

            enhanced_avg.append(run_enhanced_avg)
            enhanced_max.append(run_enhanced_max)

        # Experiments result lists for Enhanced K-Means++
        avg_delays_enhanced.append(np.mean(enhanced_avg))
        std_avg_delays_enhanced.append(np.std(enhanced_avg))
        max_delays_enhanced.append(np.mean(enhanced_max))
        std_max_delays_enhanced.append(np.std(enhanced_max))

    plot_enhanced_kmeans_experiment(
        k_values,
        avg_delays_advanced,
        max_delays_advanced,
        avg_delays_enhanced,
        std_avg_delays_enhanced,
        max_delays_enhanced,
        std_max_delays_enhanced,
        experiment_name="Advanced K-Means vs Enhanced K-Means++ (with std dev)",
        topology_name="Internet2 OS3E"
    )


