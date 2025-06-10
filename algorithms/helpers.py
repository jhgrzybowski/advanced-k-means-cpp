# === Helpers ===
# Helper functions for K-Means clustering algorithms

import networkx as nx

def compute_path_lengths(G):
    """
    Returns paths computed for the given network using Dijkstra's algorithm.
    Args:
        G (nx.Graph): Input graph.

    Returns:
        path_lengths (dict): Shortest delays path.
    """
    return dict(nx.all_pairs_dijkstra_path_length(G, weight='delay_ms'))

def compute_node_average_degree(G):
    """
    Computes the average degree of nodes in the graph, rounded to the nearest integer.

    Args:
        G (nx.Graph): The input undirected graph with delay-weighted edges.

    Returns:
        avg_degree (int): Average degree of nodes in the graph (rounded).
    """

    degrees = dict(G.degree())
    nodes = list(G.nodes())

    # Sum up degrees in the network
    total_degree_count = sum(degrees.values())

    # Count up the nodes
    num_nodes = len(nodes)

    return int(round(total_degree_count / num_nodes)) if num_nodes else 0


def normalize_metrics(node, degrees, betweenness, closeness):
    """
    Normalizes degree, betweenness, and closeness for a node to [0,1].
    """
    max_deg = max(degrees.values()) if degrees else 1
    max_bet = max(betweenness.values()) if betweenness else 1
    max_clo = max(closeness.values()) if closeness else 1
    d_norm = degrees[node] / max_deg if max_deg else 0
    b_norm = betweenness[node] / max_bet if max_bet else 0
    c_norm = closeness[node] / max_clo if max_clo else 0
    return d_norm, b_norm, c_norm


def satisfies_degree(node, degrees, avg_degree):
    """
    Checks whether a node's degree is greater than or equal to the rounded average degree.

    Args:
        node (int): Node ID.
        degrees (dict): Mapping {node: degree} for all nodes.
        avg_degree (int): Rounded average degree (minimum required for a valid center).

    Returns:
        bool: True if the node's degree >= avg_degree, otherwise False.
    """
    return degrees[node] >= avg_degree

def select_farthest_node(nodes, centers, degrees, avg_degree, path_lengths):
    """
    Selects the node (not already a center) that is farthest (in terms of minimal
    shortest path distance to any existing center) and satisfies the minimum degree constraint.

    Args:
        nodes (list): List of all node IDs in the graph.
        centers (list): List of already selected center node IDs.
        degrees (dict): Mapping {node: degree}.
        avg_degree (int): Rounded average degree for the network.
        path_lengths (dict): Nested dict {node: {target: shortest_path_length}}.

    Returns:
        int or None: Node ID of the farthest valid node, or None if none found.
    """
    farthest_node = None
    max_min_dist = -1
    for n in nodes:
        if n in centers or not satisfies_degree(n, degrees, avg_degree):
            continue
        min_dist = min(path_lengths[n][c] for c in centers)
        if min_dist > max_min_dist:
            max_min_dist = min_dist
            farthest_node = n
    return farthest_node


def select_stochastic_next_center(G, centers, rng):
    """
    Samples a new center among nodes not in centers (prefer degree >= avg_degree)
    with probability proportional to squared min distance to any center.
    """
    nodes = list(G.nodes())
    degrees = dict(G.degree())
    avg_degree = compute_node_average_degree(G)

    path_lengths = compute_path_lengths(G)

    candidates = [n for n in nodes if n not in centers and satisfies_degree(n, degrees, avg_degree)]
    if not candidates:
        # Fallback: allow any node not in centers
        candidates = [n for n in nodes if n not in centers]
        if not candidates:
            return None
    dists = []
    for n in candidates:
        min_dist = min(path_lengths[n][c] for c in centers)
        dists.append(min_dist ** 2)
    total = sum(dists)
    if total == 0:
        return rng.choice(candidates)
    chosen_idx = rng.choices(range(len(candidates)), weights=dists, k=1)[0]
    return candidates[chosen_idx]


def assign_nodes_to_centers(centers, nodes, path_lengths):
    """
    Assigns each node in the graph to the closest center (controller) based on shortest path length.

    Args:
        centers (list): List of center node IDs.
        nodes (list): List of all node IDs in the graph.
        path_lengths (dict): Nested dict {node: {target: shortest_path_length}}.

    Returns:
        dict: Mapping {center_node: set of assigned node IDs}.
    """
    clusters = {c: set() for c in centers}
    for n in nodes:
        closest_center = min(centers, key=lambda c: path_lengths[n][c])
        clusters[closest_center].add(n)
    return clusters


def update_centers(clusters, degrees, avg_degree, path_lengths):
    """
    For each cluster, selects as center the node with the minimal sum of delays to all
    other nodes in the cluster, preferring nodes with degree >= avg_degree.
    Falls back to any node in the cluster if no eligible candidates exist.

    Args:
        clusters (dict): Mapping {center_node: set of nodes in the cluster}.
        degrees (dict): Mapping {node: degree}.
        avg_degree (int): Rounded average degree for the network.
        path_lengths (dict): Nested dict {node: {target: shortest_path_length}}.

    Returns:
        list: List of node IDs to be used as updated centers for each cluster (order matches clusters.values()).
    """
    new_centers = []
    for members in clusters.values():
        eligible = [n for n in members if satisfies_degree(n, degrees, avg_degree)]
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
