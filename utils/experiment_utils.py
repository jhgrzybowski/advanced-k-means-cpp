import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from utils.data_utils import load_gml_to_delay_graph

def run_latency_experiment(
    gml_file,
    clustering_fn,
    algorithm_name,
    propagation_speed_km_per_ms,
    kmax
):
    """
    Runs latency experiments for a given topology and clustering algorithm.

    Args:
        gml_file (str): Path to the GML file.
        clustering_fn (callable): Function that performs clustering/controller placement (e.g. advanced_k_means).
            Must accept arguments (G, k) and return (controllers, clusters).
        algorithm_name (str): Name of the algorithm (used in plot labels/filenames).
        propagation_speed_km_per_ms (float): Signal propagation speed for the delay calculation.
        kmax (int): Maximum number of controllers to test.


    Saves:
        Average and maximum latency plots for all tested k (1...kmax) in 'plots/' directory.
    """
    import os
    os.makedirs("plots", exist_ok=True)

    # Load the topology
    G = load_gml_to_delay_graph(gml_file, propagation_speed_km_per_ms=propagation_speed_km_per_ms)

    k_values = list(range(1, kmax + 1))
    avg_times = []
    max_times = []

    for k in k_values:

        controllers, clusters = clustering_fn(G, k)

        # Log to console
        print(60*"#")
        print("Controllers (IDs):", controllers)
        for ctrl in controllers:
            ctrl_label = G.nodes[ctrl].get('label', str(ctrl))
            members_labels = [G.nodes[n].get('label', str(n)) for n in clusters[ctrl]]
            print(f"  Controller {ctrl} ({ctrl_label}): {members_labels}")

        # Calculate the delay from every node to its controller (shortest path delay)
        node_to_ctrl_delay = []
        for ctrl in controllers:
            for node in clusters[ctrl]:
                # If node is the controller, delay = 0
                if node == ctrl:
                    node_to_ctrl_delay.append(0.0)
                else:
                    # Dijkstra shortest path length in ms (weight='delay_ms') - L(Cj)
                    delay = nx.shortest_path_length(G, source=node, target=ctrl, weight='delay_ms')
                    node_to_ctrl_delay.append(delay)

        # ∑_(j in C) L(Cj)
        total_delay = sum(node_to_ctrl_delay)

        num_nodes = G.number_of_nodes()

        avg_latency = total_delay / num_nodes if num_nodes else 0.0
        max_latency = max(node_to_ctrl_delay) if node_to_ctrl_delay else 0.0

        print(f'for k={k}, avg_latency={avg_latency}, max_latency={max_latency}')

        avg_times.append(avg_latency)
        max_times.append(max_latency)



    # Plot average latency
    plt.figure(1, figsize=(12, 6))
    plt.plot(k_values, avg_times, 'b-o', label=algorithm_name)
    plt.xlabel('Number of Controllers')
    plt.ylabel('Average Response Time (ms)')
    plt.title(f'{algorithm_name} - Average Latency')
    plt.ylim(0, 8)
    plt.grid(True)
    plt.legend()
    ax1 = plt.gca()
    ax1.xaxis.set_major_locator(MultipleLocator(1))
    ax1.yaxis.set_major_locator(MultipleLocator(0.25))
    plt.savefig(f'plots/{algorithm_name}_average_latency.png', bbox_inches='tight')
    plt.show()
    plt.close()

    # Plot maximum latency
    plt.figure(2, figsize=(12, 6))
    plt.plot(k_values, max_times, 'r-o', label=algorithm_name)
    plt.xlabel('Number of Controllers')
    plt.ylabel('Maximum Response Time (ms)')
    plt.title(f'{algorithm_name} - Worst-case Latency')
    plt.ylim(0, 15.5)
    plt.grid(True)
    plt.legend()
    ax2 = plt.gca()
    ax2.xaxis.set_major_locator(MultipleLocator(1))
    ax2.yaxis.set_major_locator(MultipleLocator(0.5))
    plt.savefig(f'plots/{algorithm_name}_max_latency.png', bbox_inches='tight')
    plt.show()
    plt.close()
