import networkx as nx

def compute_latencies_for_experiment(G, k, controllers, clusters):
    """
    Computes average and maximum propagation latencies for controller placement experiments.

    Args:
        G (networkx.Graph): The network graph with delay weights on edges (attribute "delay_ms").
        controllers_list (list): List of controllers' lists for each k (e.g., [controllers_k1, controllers_k2, ...]).
        clusters_list (list): List of clusters' dicts for each k (e.g., [clusters_k1, clusters_k2, ...]).
            Each clusters dict: {controller: [node1, node2, ...], ...}

    Returns:
        avg_delays (list of float): Average propagation latency for each k.
        max_delays (list of float): Maximum propagation latency for each k.
    """
    avg_delays = []
    max_delays = []

    # For each experiment (value of k, i.e., number of controllers)

    delays = []
    for ctrl in controllers:
        for node in clusters[ctrl]:
            # Do not consider latency of controller (it is 0.0)
            if node != ctrl:
                # Calculate shortest path delay from node to controller
                delay = nx.shortest_path_length(G, source=node, target=ctrl, weight="delay_ms")
                delays.append(delay)
    total_delay = sum(delays)
    num_nodes = G.number_of_nodes() - k
    avg_latency = total_delay / num_nodes if num_nodes else 0.0
    max_latency = max(delays) if delays else 0.0

    avg_delays.append(avg_latency)
    max_delays.append(max_latency)

    return avg_delays, max_delays
