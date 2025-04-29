import networkx as nx


def calculate_response_times(G, controllers):
    """
    Unified method to calculate average and maximum propagation delays.

    Parameters:
        G (nx.Graph): Network graph
        controllers (list): List of controller nodes

    Returns:
        tuple: (average_delay, max_delay) in milliseconds
    """
    if not controllers:
        return float('inf'), float('inf')

    try:
        # Calculate shortest paths from all controllers
        distances = nx.multi_source_dijkstra_path_length(
            G,
            controllers,
            weight='weight'
        )

        # Exclude controller nodes from delay calculation
        node_delays = [delay for node, delay in distances.items()
                       if node not in controllers]

        if not node_delays:
            return 0.0, 0.0  # All nodes are controllers

        avg_delay = sum(node_delays) / len(node_delays)
        max_delay = max(node_delays)

        return avg_delay, max_delay

    except (nx.NetworkXNoPath, ValueError):
        return float('inf'), float('inf')