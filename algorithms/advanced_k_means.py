# === Advanced K-Means ===
# Main Algorithm 1 (Initial Center) and Algorithm 2 (Network Partitioning)

import networkx as nx

from algorithms.helpers import (
    satisfies_degree,
    select_farthest_node,
    assign_nodes_to_centers,
    update_centers,
    compute_path_lengths,
    compute_node_average_degree
)

def best_initial_center(G):
    """
    Algorithm 1: Selects the initial cluster center for Advanced K-Means.
    The node with the highest degree (and degree >= avg_degree) is chosen.
    In case of a tie (multiple nodes with maximum degree), the node with the minimum
    sum of shortest path lengths to all other nodes is selected.

    Args:
        G (nx.Graph): The input undirected graph with delay-weighted edges.

    Returns:
        int: Node ID of the selected best center.
    """

    nodes = list(G.nodes())
    degrees = dict(G.degree())
    avg_degree = compute_node_average_degree(G)

    path_lengths = compute_path_lengths(G)

    candidates = [n for n in nodes if satisfies_degree(n, degrees, avg_degree)]
    if not candidates:
        raise ValueError("No eligible node with required degree for initial center.")
    max_deg = max(degrees[n] for n in candidates)
    best = None
    min_sum_dist = float('inf')
    for n in candidates:
        if degrees[n] == max_deg:
            sum_dist = sum(path_lengths[n][m] for m in nodes)
            if sum_dist < min_sum_dist:
                min_sum_dist = sum_dist
                best = n
    return best

def advanced_k_means(G, k):
    """
    Performs the Advanced K-Means clustering for SDN controller placement.
    This implementation follows Algorithm 2 from the paper:
    "Optimizing SDN Controller to Switch Latency for Controller Placement Problem" (F. Zobary, 2024).

    The algorithm iteratively selects controller locations (centers) to minimize
    the average propagation delay between controllers and switches.
    After adding each center, a local K-Means cycle is performed for the current number of centers.

    Args:
        G (nx.Graph): Undirected graph with delay-weighted edges (attribute "delay_ms").
        k (int): Number of controllers (clusters).

    Returns:
        controllers (list): List of selected controller node ids.
        clusters (dict): Mapping from controller node id to set of assigned node ids.
    """
    nodes = list(G.nodes())
    degrees = dict(G.degree())
    avg_degree = compute_node_average_degree(G)

    path_lengths = compute_path_lengths(G)

    # Step 1: Select the first center (Algorithm 1)
    centers = [best_initial_center(G)]

    j = 2
    while j <= k:
        # Step 2: Select next center as the farthest eligible node from current centers
        new_center = select_farthest_node(
            nodes, centers, degrees, avg_degree, path_lengths
        )
        centers.append(new_center)
        # Step 3: Local K-Means cycle (assignment + center update) until convergence
        while True:
            clusters = assign_nodes_to_centers(centers, nodes, path_lengths)
            new_centers = update_centers(clusters, degrees, avg_degree, path_lengths)
            if set(new_centers) == set(centers):
                break
            centers = new_centers
        j += 1

    clusters = assign_nodes_to_centers(centers, nodes, path_lengths)
    return centers, clusters
