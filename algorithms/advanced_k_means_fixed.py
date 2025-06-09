import networkx as nx

def advanced_k_means_fixed(G, k):
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
    avg_degree = int(round(sum(degrees.values()) / len(degrees)))

    # Compute all-pairs shortest path lengths (delay in ms)
    path_lengths = dict(nx.all_pairs_dijkstra_path_length(G, weight="delay_ms"))

    def satisfies_degree(node):
        """Checks if the node satisfies the minimum degree condition."""
        return degrees[node] >= avg_degree

    def best_initial_center():
        """
        Selects the initial center:
        - node with the highest degree (>= avg_degree)
        - if tie, node with minimal sum of shortest path lengths
        """
        candidates = [n for n in nodes if satisfies_degree(n)]
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

    def select_farthest_node(centers):
        """
        Selects the node (not in centers) with the maximal minimum distance to any current center,
        satisfying the degree condition.
        """
        farthest_node = None
        max_min_dist = -1
        for n in nodes:
            if n in centers or not satisfies_degree(n):
                continue
            min_dist = min(path_lengths[n][c] for c in centers)
            if min_dist > max_min_dist:
                max_min_dist = min_dist
                farthest_node = n
        return farthest_node

    def assign_nodes_to_centers(centers):
        """
        Assigns each node to the closest center.
        Returns a dictionary: {center: set(nodes)}.
        """
        clusters = {c: set() for c in centers}
        for n in nodes:
            closest_center = min(centers, key=lambda c: path_lengths[n][c])
            clusters[closest_center].add(n)
        return clusters

    def update_centers(clusters):
        """
        For each cluster, selects the node with minimal sum of delays to other nodes in the cluster,
        satisfying the degree condition if possible.
        """
        new_centers = []
        for members in clusters.values():
            best = None
            min_sum = float('inf')
            # Prefer nodes satisfying degree condition, but fallback if none found (e.g. for small clusters)
            candidates = [n for n in members if satisfies_degree(n)]
            if not candidates:
                candidates = list(members)
            for n in candidates:
                sum_dist = sum(path_lengths[n][m] for m in members)
                if sum_dist < min_sum:
                    min_sum = sum_dist
                    best = n
            new_centers.append(best)
        return new_centers

    # === Main Advanced K-Means logic ===
    centers = [best_initial_center()]
    j = 2
    while j <= k:
        # Select the next center (farthest from current centers, satisfying degree constraint)
        new_center = select_farthest_node(centers)
        if new_center is None:
            # If no more candidates satisfy the degree constraint, allow any node not already a center
            noncenters = [n for n in nodes if n not in centers]
            if not noncenters:
                break  # All nodes used
            # Fallback: pick the node farthest from centers
            new_center = max(
                noncenters,
                key=lambda n: min(path_lengths[n][c] for c in centers)
            )
        centers.append(new_center)
        # Local K-means cycle for the current number of centers
        while True:
            clusters = assign_nodes_to_centers(centers)
            new_centers = update_centers(clusters)
            if set(new_centers) == set(centers):
                break
            centers = new_centers
        j += 1

    # Final assignment (can be omitted if clusters remain valid from last iteration)
    clusters = assign_nodes_to_centers(centers)
    return centers, clusters
