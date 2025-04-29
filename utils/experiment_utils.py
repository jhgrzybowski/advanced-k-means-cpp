import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from utils.data_utils import read_and_preprocess_gml
from utils.metrics_utils import calculate_response_times


def run_controller_experiment(algorithm, algorithm_name, file_path, max_controllers=12):
    """
    Unified experiment runner for controller placement algorithms

    Parameters:
        algorithm (callable): The algorithm function to test
        algorithm_name (str): Name of the algorithm for labeling
        file_path (str): Path to .gml file
        max_controllers (int): Maximum number of controllers to test
    """
    # Load and preprocess network
    G = read_and_preprocess_gml(file_path)

    # Prepare results storage
    k_values = []
    avg_times = []
    max_times = []

    # Run experiments for different controller counts
    for k in range(1, max_controllers + 1):
        # Get controllers using the specified algorithm
        result = algorithm(G, k)

        # Handle different return types (HDIDS vs K-Means)
        if isinstance(result, tuple):
            controllers, _ = result  # Unpack Advanced K-Means result
        else:
            controllers = result  # HDIDS direct result

        if not controllers:
            print(f"No controllers placed for k={k}")
            continue

        # Calculate response times using multi-source Dijkstra
        avg, max_ = calculate_response_times(G, controllers)

        # Store results
        k_values.append(k)
        avg_times.append(avg)
        max_times.append(max_)
        print(f"{algorithm_name} k={k}: Avg={avg:.2f}ms, Max={max_:.2f}ms")

    # Create plots in separate windows
    plt.figure(1, figsize=(12, 6))
    plt.plot(k_values, avg_times, 'b-o', label=algorithm_name)
    plt.xlabel('Number of Controllers')
    plt.ylabel('Average Response Time (ms)')
    plt.title(f'{algorithm_name} - Average Latency')
    plt.ylim(0, 5.75)
    plt.grid(True)
    plt.legend()

    ax1 = plt.gca()
    ax1.xaxis.set_major_locator(MultipleLocator(1))
    ax1.yaxis.set_major_locator(MultipleLocator(0.25))

    plt.savefig(f'plots/{algorithm_name}_average_latency.png', bbox_inches='tight')
    plt.show()
    plt.close()

    plt.figure(2, figsize=(12, 6))
    plt.plot(k_values, max_times, 'r-o', label=algorithm_name)
    plt.xlabel('Number of Controllers')
    plt.ylabel('Maximum Response Time (ms)')
    plt.title(f'{algorithm_name} - Worst-case Latency')
    plt.ylim(0, 11)
    plt.grid(True)
    plt.legend()

    ax2 = plt.gca()
    ax2.xaxis.set_major_locator(MultipleLocator(1))
    ax2.yaxis.set_major_locator(MultipleLocator(0.5))

    plt.savefig(f'plots/{algorithm_name}_max_latency.png', bbox_inches='tight')
    plt.show()
    plt.close()