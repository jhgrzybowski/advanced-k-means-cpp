# import networkx as nx
# from algorithms.helpers import _compute_path_lengths, _prepare_graph_features, _prepare_weight_normalization, _eligible_center_nodes
#
# def best_weighted_initial_center(
#     G,
#     betweenness,
#     closeness,
#     w_degree,
#     w_betweenness,
#     w_closeness,
# ):
#     """
#     Selects the first initial cluster center for the Enhanced K-Means algorithm using weighted sum of:
#     highest degree, betweenness, and closeness centrality. All metrics are normalized before weighting.
#     Only nodes with degree >= average degree are considered.
#
#     Args:
#         G (nx.Graph): The input undirected graph with delay-weighted edges.
#         betweenness (dict): Betweenness centrality {node: value}.
#         closeness (dict): Closeness centrality {node: value}.
#         w_degree (float): Weight for degree centrality.
#         w_betweenness (float): Weight for betweenness centrality.
#         w_closeness (float): Weight for closeness centrality.
#
#     Returns:
#         best_node (int): Node id of the selected best center.
#     """
#
#     # Select nodes that can be candidates for initial center
#     eligible_nodes, degrees = _prepare_graph_features(G)
#
#     # Compute all shortest paths in the network
#     path_lengths = _compute_path_lengths(G)
#
#     best_node = None
#     best_score = -float('inf')
#     best_sum = float('inf')
#
#     # For each egligible node calculate the score for being the best candidate for initial center
#     for n in eligible_nodes:
#
#         d_norm, b_norm, c_norm = _prepare_weight_normalization(G,n,betweenness, closeness)
#
#         # Score is calculated based on weighted metrics
#         score = w_degree * d_norm + w_betweenness * b_norm + w_closeness * c_norm
#         sum_dist = sum(path_lengths[n].values())
#         if (score > best_score) or (score == best_score and sum_dist < best_sum):
#             best_node = n
#             best_score = score
#             best_sum = sum_dist
#     return best_node
#
# def enhanced_k_means_fixed(G, k, rng, w_degree, w_betweenness, w_closeness):
#     """
#     Enhanced K-Means clustering for SDN controller placement.
#     After selecting the first center using a weighted centrality function,
#     the next centers are chosen using a k-means++-like stochastic method
#     (probability proportional to squared minimum distance to any current center),
#     but after each addition, a local k-means cycle is performed until convergence.
#
#     Args:
#         G (nx.Graph): The input undirected graph with delay-weighted edges.
#         k (int): Number of controllers/clusters.
#         rng (random.Random): Random number generator.
#         w_degree (float): Weight for degree centrality.
#         w_betweenness (float): Weight for betweenness centrality.
#         w_closeness (float): Weight for closeness centrality.
#
#     Returns:
#         controllers (list): List of selected controller node ids.
#         clusters (dict): Mapping from controller node id to set of assigned node ids.
#     """
#     nodes = list(G.nodes())
#     degrees = dict(G.degree())
#     avg_degree = int(round(sum(degrees.values()) / len(degrees)))
#
#     # Precompute all-pairs shortest path lengths (delays)
#     path_lengths = _compute_path_lengths(G)
#
#     betweenness = nx.betweenness_centrality(G, normalized=True, weight='delay_ms')
#     closeness = nx.closeness_centrality(G, distance='delay_ms')
#
#     def satisfies_degree(node):
#         """Check if the node has at least the average degree."""
#         return degrees[node] >= avg_degree
#
#     def sample_center_kmeanspp(centers):
#         """
#         Stochastically select a new center among nodes not in centers, degree >= avg,
#         with probability proportional to (min_dist_to_centers)^2.
#         If all probabilities are zero or no candidates, fall back to any eligible node.
#         """
#         candidates = [n for n in nodes if n not in centers and satisfies_degree(n)]
#         if not candidates:
#             # If no eligible nodes, fall back to any non-center
#             candidates = [n for n in nodes if n not in centers]
#             if not candidates:
#                 return None
#         dists = []
#         for n in candidates:
#             min_dist = min(path_lengths[n][c] for c in centers)
#             dists.append(min_dist ** 2)
#         total = sum(dists)
#         if total == 0:
#             # All distances are zero (shouldn't happen, but fallback)
#             return rng.choice(candidates)
#         # Sample using the distribution
#         chosen_idx = rng.choices(range(len(candidates)), weights=dists, k=1)[0]
#         return candidates[chosen_idx]
#
#     def assign_nodes_to_centers(centers):
#         """Assign each node to the closest center."""
#         clusters = {c: set() for c in centers}
#         for n in nodes:
#             closest_center = min(centers, key=lambda c: path_lengths[n][c])
#             clusters[closest_center].add(n)
#         return clusters
#
#     def update_centers(clusters):
#         """
#         For each cluster, select as center the node with minimal sum of delays
#         to other nodes in the cluster, preferring nodes with degree >= avg_degree.
#         """
#         new_centers = []
#         for members in clusters.values():
#             # Prefer nodes satisfying degree condition, else fallback to any in cluster
#             eligible = _eligible_center_nodes(degrees, avg_degree)
#             if not eligible:
#                 eligible = list(members)
#             best = None
#             min_sum = float('inf')
#             for n in eligible:
#                 s = sum(path_lengths[n][m] for m in members)
#                 if s < min_sum:
#                     min_sum = s
#                     best = n
#             new_centers.append(best)
#         return new_centers
#
#     # === Main Enhanced K-Means logic ===
#     centers = [best_weighted_initial_center(G, betweenness, closeness, w_degree, w_betweenness, w_closeness)]
#     j = 2
#     while j <= k:
#         # 1. Select next center via k-means++ stochastic sampling
#         next_center = sample_center_kmeanspp(centers)
#         if next_center is None:
#             break  # No more eligible nodes
#         centers.append(next_center)
#         # 2. After adding, perform local k-means until convergence
#         while True:
#             clusters = assign_nodes_to_centers(centers)
#             new_centers = update_centers(clusters)
#             if set(new_centers) == set(centers):
#                 break
#             centers = new_centers
#         j += 1
#
#     clusters = assign_nodes_to_centers(centers)
#     return centers, clusters

import networkx as nx

def enhanced_k_means_fixed(G, k, rng, w_degree, w_betweenness, w_closeness):
    """
    Enhanced K-Means clustering for SDN controller placement.
    After selecting the first center using a weighted centrality function,
    the next centers are chosen using a k-means++-like stochastic method
    (probability proportional to squared minimum distance to any current center),
    but after each addition, a local k-means cycle is performed until convergence.

    Args:
        G (nx.Graph): The input undirected graph with delay-weighted edges.
        k (int): Number of controllers/clusters.
        rng (random.Random): Random number generator.
        w_degree (float): Weight for degree centrality.
        w_betweenness (float): Weight for betweenness centrality.
        w_closeness (float): Weight for closeness centrality.

    Returns:
        controllers (list): List of selected controller node ids.
        clusters (dict): Mapping from controller node id to set of assigned node ids.
    """
    nodes = list(G.nodes())
    degrees = dict(G.degree())
    avg_degree = int(round(sum(degrees.values()) / len(degrees)))

    # Precompute all-pairs shortest path lengths (delays)
    path_lengths = dict(nx.all_pairs_dijkstra_path_length(G, weight="delay_ms"))
    betweenness = nx.betweenness_centrality(G, normalized=True, weight='delay_ms')
    closeness = nx.closeness_centrality(G, distance='delay_ms')

    def satisfies_degree(node):
        """Check if the node has at least the average degree."""
        return degrees[node] >= avg_degree

    def normalize_metrics(node):
        """Return normalized (0..1) degree, betweenness, closeness for node."""
        d_norm = degrees[node] / max(degrees.values()) if max(degrees.values()) > 0 else 0
        b_norm = betweenness[node] / max(betweenness.values()) if max(betweenness.values()) > 0 else 0
        c_norm = closeness[node] / max(closeness.values()) if max(closeness.values()) > 0 else 0
        return d_norm, b_norm, c_norm

    def best_weighted_initial_center():
        """
        Select the initial center using a weighted sum of normalized
        degree, betweenness, and closeness. Only eligible nodes (degree >= avg).
        In case of tie, choose the node with minimal sum of shortest path distances.
        """
        candidates = [n for n in nodes if satisfies_degree(n)]
        max_score = -float('inf')
        min_sum = float('inf')
        best = None
        for n in candidates:
            d_norm, b_norm, c_norm = normalize_metrics(n)
            score = w_degree * d_norm + w_betweenness * b_norm + w_closeness * c_norm
            sum_dist = sum(path_lengths[n][m] for m in nodes)
            if (score > max_score) or (score == max_score and sum_dist < min_sum):
                best = n
                max_score = score
                min_sum = sum_dist
        return best

    def sample_center_kmeanspp(centers):
        """
        Stochastically select a new center among nodes not in centers, degree >= avg,
        with probability proportional to (min_dist_to_centers)^2.
        If all probabilities are zero or no candidates, fall back to any eligible node.
        """
        candidates = [n for n in nodes if n not in centers and satisfies_degree(n)]
        if not candidates:
            # If no eligible nodes, fall back to any non-center
            candidates = [n for n in nodes if n not in centers]
            if not candidates:
                return None
        dists = []
        for n in candidates:
            min_dist = min(path_lengths[n][c] for c in centers)
            dists.append(min_dist ** 2)
        total = sum(dists)
        if total == 0:
            # All distances are zero (shouldn't happen, but fallback)
            return rng.choice(candidates)
        # Sample using the distribution
        chosen_idx = rng.choices(range(len(candidates)), weights=dists, k=1)[0]
        return candidates[chosen_idx]

    def assign_nodes_to_centers(centers):
        """Assign each node to the closest center."""
        clusters = {c: set() for c in centers}
        for n in nodes:
            closest_center = min(centers, key=lambda c: path_lengths[n][c])
            clusters[closest_center].add(n)
        return clusters

    def update_centers(clusters):
        """
        For each cluster, select as center the node with minimal sum of delays
        to other nodes in the cluster, preferring nodes with degree >= avg_degree.
        """
        new_centers = []
        for members in clusters.values():
            # Prefer nodes satisfying degree condition, else fallback to any in cluster
            eligible = [n for n in members if satisfies_degree(n)]
            if not eligible:
                eligible = list(members)
            best = None
            min_sum = float('inf')
            for n in eligible:
                s = sum(path_lengths[n][m] for m in members)
                if s < min_sum:
                    min_sum = s
                    best = n
            new_centers.append(best)
        return new_centers

    # === Main Enhanced K-Means logic ===
    centers = [best_weighted_initial_center()]
    j = 2
    while j <= k:
        # 1. Select next center via k-means++ stochastic sampling
        next_center = sample_center_kmeanspp(centers)
        if next_center is None:
            break  # No more eligible nodes
        centers.append(next_center)
        # 2. After adding, perform local k-means until convergence
        while True:
            clusters = assign_nodes_to_centers(centers)
            new_centers = update_centers(clusters)
            if set(new_centers) == set(centers):
                break
            centers = new_centers
        j += 1

    clusters = assign_nodes_to_centers(centers)
    return centers, clusters

