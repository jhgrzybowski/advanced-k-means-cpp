# === Enhanced K-Means++ ===
# Main Algorithm 1 (Weighted Initial Center) and Algorithm 2 (Network Partitioning with stochastic cluster selection)

import networkx as nx

from algorithms.helpers import (
    normalize_metrics,
    satisfies_degree,
    select_stochastic_next_center,
    assign_nodes_to_centers,
    update_centers,
    compute_path_lengths,
    compute_node_average_degree
)

def best_weighted_initial_center(
    G,
    betweenness,
    closeness,
    w_degree,
    w_betweenness,
    w_closeness,
):
    """
    Algorithm 1: Selects the initial cluster center for Enhanced K-Means using a weighted sum
    of normalized degree, betweenness, and closeness centrality. Only nodes with degree
    greater than or equal to the rounded average degree are considered as eligible candidates.

    In case of ties (same score), the node with the smallest sum of shortest path distances
    to all other nodes is chosen.

    Args:
        G (nx.Graph): The input undirected graph with delay-weighted edges.
        betweenness (dict): Mapping {node: betweenness centrality (float)}.
        closeness (dict): Mapping {node: closeness centrality (float)}.
        w_degree (float): Weight for degree centrality (normalized).
        w_betweenness (float): Weight for betweenness centrality (normalized).
        w_closeness (float): Weight for closeness centrality (normalized).

    Returns:
        int: Node ID of the selected best center.
    """
    nodes = list(G.nodes())
    degrees = dict(G.degree())
    avg_degree = compute_node_average_degree(G)

    path_lengths = compute_path_lengths(G)

    candidates = [n for n in nodes if satisfies_degree(n, degrees, avg_degree)]
    max_score = -float('inf')
    min_sum = float('inf')
    best = None
    for n in candidates:
        d_norm, b_norm, c_norm = normalize_metrics(n, degrees, betweenness, closeness)
        score = w_degree * d_norm + w_betweenness * b_norm + w_closeness * c_norm
        sum_dist = sum(path_lengths[n][m] for m in nodes)
        if (score > max_score) or (score == max_score and sum_dist < min_sum):
            best = n
            max_score = score
            min_sum = sum_dist
    return best

def enhanced_k_means(G, k, rng, w_degree, w_betweenness, w_closeness):
    """
    Algorithm 2: Enhanced K-Means clustering for SDN controller placement.
    Partitions the graph into k clusters by selecting controller nodes (centers)
    in a way that aims to minimize the average propagation delay (latency)
    between controllers and switches.

    After each new center is added (using a stochastic k-means++ style rule),
    a local k-means cycle (assignment and centroid update) is performed for the current
    set of centers, until convergence.

    Args:
        G (nx.Graph): The input undirected graph with delay-weighted edges (attribute: "delay_ms").
        k (int): Desired number of clusters/controllers.
        rng (random.Random): Random number generator for stochastic sampling.
        w_degree (float): Weight for degree centrality in initial center selection.
        w_betweenness (float): Weight for betweenness centrality in initial center selection.
        w_closeness (float): Weight for closeness centrality in initial center selection.

    Returns:
        tuple:
            - controllers (list): List of selected controller node IDs (cluster centers).
            - clusters (dict): Mapping from controller node ID to set of assigned node IDs in its cluster.
    """

    nodes = list(G.nodes())
    degrees = dict(G.degree())
    avg_degree = compute_node_average_degree(G)

    path_lengths = compute_path_lengths(G)
    betweenness = nx.betweenness_centrality(G, normalized=True, weight='delay_ms')
    closeness = nx.closeness_centrality(G, distance='delay_ms')

    # Step 1: Select the first center using weighted centrality (Algorithm 1)
    centers = [best_weighted_initial_center(
        G, betweenness, closeness,
        w_degree, w_betweenness, w_closeness
    )]

    j = 2
    while j <= k:
        # Step 2: Select next center using k-means++ stochastic rule with degree constraint
        next_center = select_stochastic_next_center(G, centers, rng)
        if next_center is None:
            break
        centers.append(next_center)
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
