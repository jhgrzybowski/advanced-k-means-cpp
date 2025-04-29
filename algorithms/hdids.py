import matplotlib.pyplot as plt
from functools import partial
from utils.data_utils import *

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


def hdids(G, k):
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