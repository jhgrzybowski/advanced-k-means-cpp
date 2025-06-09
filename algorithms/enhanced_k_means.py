import networkx as nx
from utils.data_utils import *
from algorithms.helpers import _compute_path_lengths, _prepare_graph_features, _prepare_weight_normalization


def best_weighted_initial_center(
    G,
    betweenness,
    closeness,
    w_degree,
    w_betweenness,
    w_closeness,
):
    """
    Selects the first initial cluster center for the Enhanced K-Means algorithm using weighted sum of:
    highest degree, betweenness, and closeness centrality. All metrics are normalized before weighting.
    Only nodes with degree >= average degree are considered.

    Args:
        G (nx.Graph): The input undirected graph with delay-weighted edges.
        betweenness (dict): Betweenness centrality {node: value}.
        closeness (dict): Closeness centrality {node: value}.
        w_degree (float): Weight for degree centrality.
        w_betweenness (float): Weight for betweenness centrality.
        w_closeness (float): Weight for closeness centrality.

    Returns:
        best_node (int): Node id of the selected best center.
    """

    # Select nodes that can be candidates for initial center
    eligible_nodes, degrees = _prepare_graph_features(G)

    # Compute all shortest paths in the network
    path_lengths = _compute_path_lengths(G)

    best_node = None
    best_score = -float('inf')
    best_sum = float('inf')

    # For each egligible node calculate the score for being the best candidate for initial center
    for n in eligible_nodes:

        d_norm, b_norm, c_norm = _prepare_weight_normalization(G,n,betweenness, closeness)

        # Score is calculated based on weighted metrics
        score = w_degree * d_norm + w_betweenness * b_norm + w_closeness * c_norm
        sum_dist = sum(path_lengths[n].values())
        if (score > best_score) or (score == best_score and sum_dist < best_sum):
            best_node = n
            best_score = score
            best_sum = sum_dist
    return best_node

def enhanced_k_means(G, k, rng, w_degree, w_betweenness, w_closeness):
    """
    Enhanced K-Means clustering for SDN controller placement with degree, betweenness, and closeness weight system.
    It partitions the network into k clusters and selects k controller nodes to minimize average propagation delay.
    The initial controller is selected using select_first_initial_center(), but with full centrality weighting.

    Args:
        G (nx.Graph): The input undirected graph with delay-weighted edges.
        k (int): Number of controllers/clusters.
        rng (random.Random(seed)): Random number generator.
        w_degree (float): Weight for degree centrality (default: 1.0).
        w_betweenness (float): Weight for betweenness centrality (default: 1.0).
        w_closeness (float): Weight for closeness centrality (default: 1.0).

    Returns:
        controllers (list): List of selected controller node ids.
        clusters (dict): Mapping from controller node id to set of assigned node ids.
    """
    nodes = list(G.nodes())
    path_lengths = _compute_path_lengths(G)


    betweenness = nx.betweenness_centrality(G, normalized=True, weight='delay_ms')
    closeness = nx.closeness_centrality(G, distance='delay_ms')

    # Step 1: Select the first center using weighted centrality
    centers = [best_weighted_initial_center(
        G, betweenness, closeness,
        w_degree,
        w_betweenness,
        w_closeness
    )]

    # Step 2: Select next centers based on k-means++
    while len(centers) < k:
        # For each node, compute squared min distance to any center
        dists = []
        for node in nodes:
            if node in centers:
                dists.append(0)
            else:
                min_dist = min(path_lengths[node][c] for c in centers)
                dists.append(min_dist ** 2)
        total = sum(dists)
        probs = [d / total if total > 0 else 0 for d in dists]
        # Randomly pick the next center
        next_center = rng.choices(nodes, weights=probs, k=1)[0]
        if next_center not in centers:
            centers.append(next_center)

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
