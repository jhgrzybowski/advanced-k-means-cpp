import networkx as nx
from algorithms.helpers import _compute_path_lengths, _prepare_graph_features

def best_initial_center(G):
    """
    Selects the first initial cluster center for the Advanced K-Means algorithm.
    The node with the highest degree is preferred; in case of a tie, the one with the minimal sum of shortest path delays is chosen.
    Only nodes with degree >= average degree are considered.

    Args:
        G (nx.Graph): The input undirected graph with delay-weighted edges.

    Returns:
        best_node (int): Node id of the selected best center.
    """

    # Select nodes that can be candidates for initial center
    eligible_nodes, degrees = _prepare_graph_features(G)

    # Compute all shortest paths in the network
    path_lengths = _compute_path_lengths(G)

    best_node = None
    best_degree = -1
    best_sum = float('inf')
    for n in eligible_nodes:
        degree = degrees[n]
        sum_dist = sum(path_lengths[n].values())
        if (degree > best_degree) or (degree == best_degree and sum_dist < best_sum):
            best_node = n
            best_degree = degree
            best_sum = sum_dist
    return best_node

def advanced_k_means(G, k):
    """
    Performs the Advanced K-Means clustering for SDN controller placement.
    It partitions the network into k clusters and selects k controller nodes to minimize average propagation delay.
    The initial controller is selected using best_initial_center().

    Args:
        G (nx.Graph): The input undirected graph with delay-weighted edges.
        k (int): Number of controllers/clusters.

    Returns:
        controllers (list): List of selected controller node ids.
        clusters (dict): Mapping from controller node id to set of assigned node ids.
    """
    nodes = list(G.nodes())

    # Step 1: Select the first center
    centers = [best_initial_center(G)]

    # Step 2: Select next centers (farthest from current centers)
    path_lengths = _compute_path_lengths(G)
    while len(centers) < k:
        # For each node not already a center, find min dist to any current center
        farthest_node = None
        max_min_dist = -1
        for node in nodes:
            if node in centers:
                continue
            min_dist = min(path_lengths[node][c] for c in centers)
            if min_dist > max_min_dist:
                farthest_node = node
                max_min_dist = min_dist
        centers.append(farthest_node)

    # Step 3: Assign nodes to the closest center
    changed = True
    while changed:
        clusters = {c: set() for c in centers}
        for node in nodes:
            closest_center = min(centers, key=lambda c: path_lengths[node][c])
            clusters[closest_center].add(node)
        # Step 4: Update centers in each cluster to node with minimal sum delay in cluster
        new_centers = []
        for c, members in clusters.items():
            min_sum = float('inf')
            best_node = None
            for n in members:
                s = sum(path_lengths[n][m] for m in members)
                if s < min_sum:
                    min_sum = s
                    best_node = n
            new_centers.append(best_node)
        # Step 5: Check for convergence
        if set(new_centers) == set(centers):
            break
        centers = new_centers
    return centers, clusters