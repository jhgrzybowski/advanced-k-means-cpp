import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
plt.rcParams['font.family'] = 'Arial'

def plot_latency_results(
    k_values,
    avg_latencies,
    max_latencies,
    clustering_fns,
    experiment_name,
    topology_name,
    output_dir="plots"
):
    """
    Plots and saves average and maximum latency results for different algorithms.

    Args:
        k_values (list): List of k (number of controllers).
        avg_latencies (dict): {algorithm_name: list of avg latency}.
        max_latencies (dict): {algorithm_name: list of max latency}.
        clustering_fns (dict): {algorithm_name: clustering function} (just for iterating, not used directly).
        experiment_name (str): High-level experiment name.
        topology_name (str): Name of the topology file/experiment.
        output_dir (str): Where to save plots.
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    color_map = {
        'advanced_k_means': '#003366',    # Pantone 540C
        'enhanced_k_means': '#C8102E'     # Pantone 1797C
    }
    marker_map = {
        'advanced_k_means': 'o',
        'enhanced_k_means': 's'
    }

    # Construct experiment info string for file naming
    info = [
        f"{experiment_name}",
        f"{topology_name}",
    ]
    info_str = "__".join(info)

    # Average latency plot
    avg_filename = f"{output_dir}/Average_Latency_{info_str}.png"
    plt.figure(figsize=(12, 6))
    for name in clustering_fns:
        plt.plot(
            k_values,
            avg_latencies[name],
            label=name.replace('_', ' ').title(),
            color=color_map.get(name, None),
            marker=marker_map.get(name, None)
        )
    plt.xlabel('Number of Controllers (K)')
    plt.ylabel('Average Response Time [ms]')
    plt.title(f'Average Latency – {experiment_name} – Topology: {topology_name}')
    plt.ylim(0, 8)
    plt.grid(True)
    plt.legend()
    ax1 = plt.gca()
    ax1.xaxis.set_major_locator(MultipleLocator(1))
    ax1.yaxis.set_major_locator(MultipleLocator(0.25))
    plt.savefig(avg_filename, bbox_inches='tight')
    print(f"Plot saved: {avg_filename}")
    plt.show()
    plt.close()

    # Maximum latency plot
    max_filename = f"{output_dir}/Max_Latency_{info_str}.png"
    plt.figure(figsize=(12, 6))
    for name in clustering_fns:
        plt.plot(
            k_values,
            max_latencies[name],
            label=name.replace('_', ' ').title(),
            color=color_map.get(name, None),
            marker=marker_map.get(name, None)
        )
    plt.xlabel('Number of Controllers (K)')
    plt.ylabel('Maximum Response Time [ms]')
    plt.title(f'Maximum Latency – {experiment_name} – Topology: {topology_name}')
    plt.ylim(0, 15.5)
    plt.grid(True)
    plt.legend()
    ax2 = plt.gca()
    ax2.xaxis.set_major_locator(MultipleLocator(1))
    ax2.yaxis.set_major_locator(MultipleLocator(0.5))
    plt.savefig(max_filename, bbox_inches='tight')
    print(f"Plot saved: {max_filename}")
    plt.show()
    plt.close()
