import networkx as nx
from utils.data_utils import *

def _best_center_candidate(
    eligible_nodes,
    degrees,
    betweenness,
    closeness,
    path_lengths,
    w_degree,
    w_betweenness,
    w_closeness
):
    """
    Helper function to select the best center candidate among eligible nodes using weighted sum of degree, betweenness, and closeness centrality.
    All metrics are normalized before weighting.

    Args:
        eligible_nodes (list): List of eligible node ids.
        degrees (dict): Node degrees {node: degree}.
        betweenness (dict): Betweenness centrality {node: value}.
        closeness (dict): Closeness centrality {node: value}.
        path_lengths (dict): Shortest path delays {node: {target: delay}}.
        w_degree (float): Weight for degree centrality (default: 1.0).
        w_betweenness (float): Weight for betweenness centrality (default: 1.0).
        w_closeness (float): Weight for closeness centrality (default: 1.0).

    Returns:
        best_node (int): Node id of the selected best center.
    """
    deg_values = [degrees[n] for n in eligible_nodes]
    betw_values = [betweenness[n] for n in eligible_nodes]
    close_values = [closeness[n] for n in eligible_nodes]

    def norm(val, vmin, vmax):
        return 0 if vmax == vmin else (val - vmin) / (vmax - vmin)

    deg_min, deg_max = min(deg_values), max(deg_values)
    betw_min, betw_max = min(betw_values), max(betw_values)
    close_min, close_max = min(close_values), max(close_values)

    best_node = None
    best_score = -float('inf')
    best_sum = float('inf')

    for n in eligible_nodes:
        d_norm = norm(degrees[n], deg_min, deg_max)
        b_norm = norm(betweenness[n], betw_min, betw_max)
        c_norm = norm(closeness[n], close_min, close_max)
        score = w_degree * d_norm + w_betweenness * b_norm + w_closeness * c_norm
        sum_dist = sum(path_lengths[n].values())
        if (score > best_score) or (score == best_score and sum_dist < best_sum):
            best_node = n
            best_score = score
            best_sum = sum_dist
    return best_node

def _eligible_center_nodes(degrees, avg_degree):
    """
    Helper function to filter eligible nodes for controller placement.
    Only nodes with degree >= average degree are considered.

    Args:
        degrees (dict): Node degrees {node: degree}.
        avg_degree (int): Average node degree (rounded).
    Returns:
        eligible_nodes (list): List of eligible node ids.
    """
    eligible_nodes = [n for n, d in degrees.items() if d >= avg_degree]
    if not eligible_nodes:
        raise ValueError("No eligible nodes found with degree >= average degree.")
    return eligible_nodes

def _compute_path_lengths(G):
    """
    Returns paths computed for the given network using Dijkstra's algorithm.
    Args:
        G (nx.Graph): Input graph.

    Returns:
        path_lengths (dict): Shortest delays path.
    """
    return dict(nx.all_pairs_dijkstra_path_length(G, weight='delay_ms'))

def select_first_initial_center(
    G,
    degrees,
    betweenness,
    closeness,
    path_lengths,
    w_degree,
    w_betweenness,
    w_closeness
):
    """
    Selects the first initial cluster center (controller location) for the Enhanced K-Means algorithm
    based on weighted sum of degree, betweenness and closeness centrality (with normalization).
    Only nodes with degree >= average degree are considered.

    Args:
        G (nx.Graph): The input undirected graph with delay-weighted edges.
        degrees (dict): Node degrees {node: degree}.
        betweenness (dict): Betweenness centrality {node: value}.
        closeness (dict): Closeness centrality {node: value}.
        path_lengths (dict): Shortest path delays {node: {target: delay}}.
        w_degree (float): Weight for degree centrality (default: 1.0).
        w_betweenness (float): Weight for betweenness centrality (default: 1.0).
        w_closeness (float): Weight for closeness centrality (default: 1.0).

    Returns:
        first_center (int): Node id of the selected initial center.
    """
    total_degree = sum(degrees.values())
    num_nodes = len(degrees)
    avg_degree = int(round(total_degree / num_nodes)) if num_nodes else 0
    eligible_nodes = _eligible_center_nodes(degrees, avg_degree)
    return _best_center_candidate(
        eligible_nodes,
        degrees,
        betweenness,
        closeness,
        path_lengths,
        w_degree=w_degree,
        w_betweenness=w_betweenness,
        w_closeness=w_closeness
    )

def enhanced_k_means(G, k, w_degree, w_betweenness, w_closeness):
    """
    Enhanced K-Means clustering for SDN controller placement with degree, betweenness, and closeness weight system.
    It partitions the network into k clusters and selects k controller nodes to minimize average propagation delay.
    The initial controller is selected using select_first_initial_center(), but with full centrality weighting.

    Args:
        G (nx.Graph): The input undirected graph with delay-weighted edges.
        k (int): Number of controllers/clusters.
        w_degree (float): Weight for degree centrality (default: 1.0).
        w_betweenness (float): Weight for betweenness centrality (default: 1.0).
        w_closeness (float): Weight for closeness centrality (default: 1.0).

    Returns:
        controllers (list): List of selected controller node ids.
        clusters (dict): Mapping from controller node id to set of assigned node ids.
    """
    nodes = list(G.nodes())
    path_lengths = _compute_path_lengths(G)
    degrees = dict(G.degree())
    betweenness = nx.betweenness_centrality(G, normalized=True, weight='delay_ms')
    closeness = nx.closeness_centrality(G, distance='delay_ms')

    # Step 1: Select the first center using weighted centrality
    centers = [select_first_initial_center(
        G, degrees, betweenness, closeness, path_lengths,
        w_degree=w_degree,
        w_betweenness=w_betweenness,
        w_closeness=w_closeness
    )]

    # Step 2: Select next centers (farthest from current centers)
    while len(centers) < k:
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
