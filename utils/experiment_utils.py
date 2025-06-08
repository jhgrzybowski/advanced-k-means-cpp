import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from utils.data_utils import load_gml_to_delay_graph

def run_latency_experiment_compare(
    gml_file,
    clustering_fns,        # Dict: {alg_name: clustering_fn}
    propagation_speed_km_per_ms,
    kmax,
    enhanced_k_means_kwargs=None,  # dodatkowe argumenty do enhanced_k_means
):
    """
    Runs latency experiments for a given topology and multiple clustering algorithms.
    Args:
        gml_file (str): Path to the GML file.
        clustering_fns (dict): {algorithm_name: clustering_fn}
        propagation_speed_km_per_ms (float): Signal propagation speed.
        kmax (int): Maximum number of controllers to test.
        enhanced_k_means_kwargs (dict): Optional, extra kwargs for enhanced_k_means.
    Saves:
        Comparison plots for average and max latency.
    """
    import os
    os.makedirs("plots", exist_ok=True)

    G = load_gml_to_delay_graph(gml_file, propagation_speed_km_per_ms=propagation_speed_km_per_ms)
    k_values = list(range(1, kmax + 1))

    color_map = {
        'advanced_k_means': 'b',
        'enhanced_k_means': 'g',
        # Dodaj kolejne jeśli trzeba
    }
    marker_map = {
        'advanced_k_means': 'o',
        'enhanced_k_means': 's',
    }

    avg_latencies = {}
    max_latencies = {}

    for name, fn in clustering_fns.items():
        avg_times = []
        max_times = []

        print(80 * "=")
        print(f"## {name.upper()} ##")
        for k in k_values:
            if name == "enhanced_k_means":
                kwargs = enhanced_k_means_kwargs or {}
                controllers, clusters = fn(G, k, **kwargs)
            else:
                controllers, clusters = fn(G, k)

            # --- PRINT Kontrolery i klastry ---
            if(k==5):
                print(60 * "#")
                print(f"K={k}: Controllers (IDs): {controllers}")
                for ctrl in controllers:
                    ctrl_label = G.nodes[ctrl].get('label', str(ctrl))
                    members_labels = [G.nodes[n].get('label', str(n)) for n in clusters[ctrl]]
                    print(f"  Controller {ctrl} ({ctrl_label}): {members_labels}")

        print(f"method name: {name}")
        for k in k_values:
            # Enhanced k-means wymaga dodatkowych argumentów (wagi centralności)
            if name == "enhanced_k_means":
                # Przekaż argumenty wag jeśli są podane
                kwargs = enhanced_k_means_kwargs or {}
                controllers, clusters = fn(G, k, **kwargs)
            else:
                controllers, clusters = fn(G, k)

            # Obliczenia jak poprzednio
            node_to_ctrl_delay = []
            for ctrl in controllers:
                for node in clusters[ctrl]:
                    if node != ctrl:
                        delay = nx.shortest_path_length(G, source=node, target=ctrl, weight='delay_ms')
                        node_to_ctrl_delay.append(delay)
            total_delay = sum(node_to_ctrl_delay)
            num_nodes = G.number_of_nodes()
            avg_latency = total_delay / num_nodes if num_nodes else 0.0
            max_latency = max(node_to_ctrl_delay) if node_to_ctrl_delay else 0.0
            avg_times.append(avg_latency)
            max_times.append(max_latency)

            # --- PRINT Opóźnienia ---
            if(k==5):
                print(f"  [k={k}] avg_latency = {avg_latency:.4f}, max_latency = {max_latency:.4f}")

        print(80 * "=")
        avg_latencies[name] = avg_times
        max_latencies[name] = max_times


    # Plot average latency
    plt.figure(1, figsize=(12, 6))
    for name in clustering_fns:
        plt.plot(
            k_values,
            avg_latencies[name],
            label=name,
            color=color_map.get(name, None),
            marker=marker_map.get(name, None)
        )
    plt.xlabel('Number of Controllers')
    plt.ylabel('Average Response Time (ms)')
    plt.title('Average Latency Comparison')
    plt.ylim(0, 8)
    plt.grid(True)
    plt.legend()
    ax1 = plt.gca()
    ax1.xaxis.set_major_locator(MultipleLocator(1))
    ax1.yaxis.set_major_locator(MultipleLocator(0.25))
    plt.savefig('plots/average_latency_comparison.png', bbox_inches='tight')
    plt.show()
    plt.close()

    # Plot max latency
    plt.figure(2, figsize=(12, 6))
    for name in clustering_fns:
        plt.plot(
            k_values,
            max_latencies[name],
            label=name,
            color=color_map.get(name, None),
            marker=marker_map.get(name, None)
        )
    plt.xlabel('Number of Controllers')
    plt.ylabel('Maximum Response Time (ms)')
    plt.title('Maximum Latency Comparison')
    plt.ylim(0, 15.5)
    plt.grid(True)
    plt.legend()
    ax2 = plt.gca()
    ax2.xaxis.set_major_locator(MultipleLocator(1))
    ax2.yaxis.set_major_locator(MultipleLocator(0.5))
    plt.savefig('plots/max_latency_comparison.png', bbox_inches='tight')
    plt.show()
    plt.close()
