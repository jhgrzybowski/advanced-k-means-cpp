import networkx as nx
import math
import matplotlib.pyplot as plt
from functools import partial


def read_and_preprocess_gml(file_path):
    """
    Reads a .gml file and preprocesses the graph by calculating edge weights based on propagation delay.

    Parameters:
        file_path (str): Path to the .gml file.

    Returns:
        nx.Graph: Preprocessed undirected graph with edge weights as propagation delay (ms).
    """
    G = nx.read_gml(file_path)
    G = G.to_undirected()

    for u, v in G.edges():
        u_data = G.nodes[u]
        v_data = G.nodes[v]

        # Extract coordinates using 'lon'/'lat' keys
        lon1 = u_data.get('lon', 0.0)
        lat1 = u_data.get('lat', 0.0)
        lon2 = v_data.get('lon', 0.0)
        lat2 = v_data.get('lat', 0.0)

        # Haversine formula
        R = 6371000  # Earth radius in meters
        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        delta_phi = math.radians(lat2 - lat1)
        delta_lambda = math.radians(lon2 - lon1)

        a = (math.sin(delta_phi / 2.0) ** 2 +
             math.cos(phi1) * math.cos(phi2) *
             math.sin(delta_lambda / 2.0) ** 2)

        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        distance = R * c

        # Propagation delay in milliseconds
        G[u][v]['weight'] = (distance / 3e8) * 1000  # Speed of light

    return G


def haversine_heuristic(u, v, G):
    """
    Heuristic function for A* algorithm using Haversine distance between nodes.

    Parameters:
        u, v (nodes): Nodes to calculate heuristic between.
        G (nx.Graph): Network graph.

    Returns:
        float: Estimated propagation delay (ms) between u and v.
    """
    u_data = G.nodes[u]
    v_data = G.nodes[v]

    lon1 = u_data.get('lon', 0.0)
    lat1 = u_data.get('lat', 0.0)
    lon2 = v_data.get('lon', 0.0)
    lat2 = v_data.get('lat', 0.0)

    R = 6371000
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = (math.sin(delta_phi / 2.0) ** 2 +
         math.cos(phi1) * math.cos(phi2) *
         math.sin(delta_lambda / 2.0) ** 2)

    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return (R * c / 3e8) * 1000  # ms


def compute_total_distance(G, node, heuristic):
    """
    Computes the sum of shortest path lengths from a node to all other nodes using A*.

    Parameters:
        G (nx.Graph): Network graph.
        node (node): Source node.
        heuristic (function): Heuristic function for A*.

    Returns:
        float: Total shortest path length from node to all other nodes.
    """
    total = 0.0
    for target in G.nodes():
        if node == target:
            continue
        try:
            path_length = nx.astar_path_length(G, node, target, heuristic=heuristic, weight='weight')
            total += path_length
        except nx.NetworkXNoPath:
            pass  # Handle disconnected graphs if necessary
    return total


def hdids_algorithm(G, k):
    """
    HDIDS algorithm to select k controllers based on high degree and independent dominating set.

    Parameters:
        G (nx.Graph): Preprocessed network graph.
        k (int): Number of controllers to place.

    Returns:
        list: Selected controller nodes.
    """
    C = []  # Controller set
    S = set(G.nodes())
    heuristic = partial(haversine_heuristic, G=G)

    while len(C) < k and S:
        # Find undominated nodes in S
        S_prime = [node for node in S
                   if node not in C and
                   not any(nbr in C for nbr in G.neighbors(node))]

        if not S_prime:
            break  # All remaining nodes dominated

        # Select max degree node in S_prime
        max_degree = max(G.degree(n) for n in S_prime)
        candidates = [n for n in S_prime if G.degree(n) == max_degree]

        # Resolve ties using total distance
        if len(candidates) == 1:
            selected = candidates[0]
        else:
            min_total = float('inf')
            selected = None
            for candidate in candidates:
                total = compute_total_distance(G, candidate, heuristic)
                if total < min_total:
                    min_total = total
                    selected = candidate

        C.append(selected)
        S.remove(selected)

    return C


def calculate_response_times(G, controllers):
    """Calculate average and maximal response times."""
    if not controllers:
        return float('inf'), float('inf')

    # Multi-source shortest paths
    distances = nx.multi_source_dijkstra_path_length(G, controllers,
                                                     weight='weight')
    if not distances:
        return float('inf'), float('inf')

    min_distances = list(distances.values())
    return sum(min_distances) / len(min_distances), max(min_distances)


def run_experiments(file_path, max_controllers=12):
    """Run full experiment suite and generate plots."""
    G = read_and_preprocess_gml(file_path)

    k_values = []
    avg_times = []
    max_times = []

    for k in range(1, max_controllers + 1):
        controllers = hdids_algorithm(G, k)
        print(controllers)
        if not controllers:
            print(f"No controllers placed for k={k}")
            continue

        avg_rt, max_rt = calculate_response_times(G, controllers)
        k_values.append(k)
        avg_times.append(avg_rt)
        max_times.append(max_rt)
        print(f"k={k}: Avg={avg_rt:.2f}ms, Max={max_rt:.2f}ms")

    # Create plots
    plt.figure(figsize=(10, 5))
    plt.plot(k_values, avg_times, 'b-o', label='HDIDS')
    plt.xlabel('Number of Controllers')
    plt.ylabel('Average Response Time (ms)')
    plt.title('Average Response Time vs Controller Count')
    plt.grid(True)
    plt.legend()
    plt.show()
    # plt.savefig('average_response.png')
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(k_values, max_times, 'r-o', label='HDIDS')
    plt.xlabel('Number of Controllers')
    plt.ylabel('Maximal Response Time (ms)')
    plt.title('Worst-case Response Time vs Controller Count')
    plt.grid(True)
    plt.legend()
    plt.show()
    # plt.savefig('max_response.png')
    plt.close()


if __name__ == "__main__":
    topology_name = 'Internet2_OS3E_topology.gml'
    run_experiments(topology_name, max_controllers=12)