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

        # Display selected controllers
        print(f"[ {algorithm_name} ] Chosen controllers: {controllers}")

        # Calculate response times using multi-source Dijkstra
        avg, max_ = calculate_response_times(G, controllers)

        # Store results
        k_values.append(k)
        avg_times.append(avg)
        max_times.append(max_)
        print(f"[ {algorithm_name} ] k={k}: Avg={avg:.2f}ms, Max={max_:.2f}ms")

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


def run_comparison_experiment(algorithms, topology_name, file_path, min_controllers, max_controllers=12):
    """
    Compare multiple algorithms and plot their results on shared figures.

    Parameters:
        algorithms (list of tuples): [(algorithm_func, algorithm_name), ...]
        file_path (str): Path to .gml file
        max_controllers (int): Maximum number of controllers to test
    """
    # Load and preprocess network once
    G = read_and_preprocess_gml(file_path)

    # Prepare results storage
    results = {}
    k_values = list(range(min_controllers, max_controllers + 1))

    print(f"[ ----------------------- ]")
    # Run experiments for all algorithms
    for algorithm, name in algorithms:
        avg_times = []
        max_times = []

        for k in k_values:
            # Get controllers using the specified algorithm
            result = algorithm(G, k)

            # Handle different return types
            if isinstance(result, tuple):
                controllers, _ = result  # Unpack K-Means result
            else:
                controllers = result  # HDIDS direct result



            # Calculate response times
            avg, max_ = calculate_response_times(G, controllers)

            avg_times.append(avg)
            max_times.append(max_)
            print(f"[ {name} ] k={k}: Avg={avg:.2f}ms, Max={max_:.2f}ms")
            print(f"[ {name} ] Chosen controllers: {controllers}")
            print(f"[ ----------------------- ]")

        results[name] = {
            'avg': avg_times,
            'max': max_times
        }

    # Create comparison plots
    colors = ['b', 'r', 'g', 'k']
    markers = ['o', 's', '^', 'D', 'v', '<']
    linestyles = ['-', '--', '-.', ':']

    # Average delay comparison
    plt.figure(figsize=(10, 5))
    for idx, (name, data) in enumerate(results.items()):
        plt.plot(k_values, data['avg'],
                 color=colors[idx % len(colors)],
                 marker=markers[idx % len(markers)],
                 linestyle=linestyles[idx % len(linestyles)],
                 label=name)

    plt.xlabel('Number of Controllers')
    plt.ylabel('Average Response Time (ms)')
    plt.title(f'Average Latency Comparison - {topology_name}')
    plt.grid(True)
    plt.legend()

    ax1 = plt.gca()
    ax1.xaxis.set_major_locator(MultipleLocator(1))
    ax1.yaxis.set_major_locator(MultipleLocator(0.25))

    plt.savefig(f'plots/Average_latency_comparison_{topology_name}.png', bbox_inches='tight')
    plt.show()
    plt.close()

    # Maximum delay comparison
    plt.figure(figsize=(10, 5))
    for idx, (name, data) in enumerate(results.items()):
        plt.plot(k_values, data['max'],
                 color=colors[idx+2],
                 marker=markers[idx % len(markers)],
                 linestyle=linestyles[idx % len(linestyles)],
                 label=name)

    plt.xlabel('Number of Controllers')
    plt.ylabel('Maximum Response Time (ms)')
    plt.title(f'Worst-case Latency Comparison - {topology_name}')
    plt.grid(True)
    plt.legend()

    ax2 = plt.gca()
    ax2.xaxis.set_major_locator(MultipleLocator(1))
    ax2.yaxis.set_major_locator(MultipleLocator(0.5))

    plt.savefig(f'plots/Max_latency_comparison_{topology_name}.png', bbox_inches='tight')
    plt.show()
    plt.close()