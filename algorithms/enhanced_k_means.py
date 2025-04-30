import networkx as nx
import numpy as np
from utils.data_utils import read_and_preprocess_gml, compute_shortest_path_distances
from utils.metrics_utils import calculate_response_times


def enhanced_initial_controllers(G, k):
    """Hybrid initialization using multiple centrality metrics"""
    # Calculate average node degree
    degrees = dict(G.degree())
    avg_degree = np.mean(list(degrees.values()))
    rho = round(avg_degree)

    # Calculate centrality metrics
    betweenness = nx.betweenness_centrality(G, normalized=True)
    closeness = nx.closeness_centrality(G)

    # Create composite scores
    composite_scores = {}
    for node in G.nodes():
        if degrees[node] < rho:
            continue  # Skip nodes below average degree

        score = (degrees[node] * 0.4 +
                 betweenness[node] * 0.3 +
                 closeness[node] * 0.3)
        composite_scores[node] = score

    # Handle case with insufficient qualified nodes
    valid_nodes = sorted(composite_scores.keys(),
                         key=lambda x: composite_scores[x], reverse=True)

    if len(valid_nodes) < k:
        print(f"Warning: Only {len(valid_nodes)} nodes meet degree threshold")
        valid_nodes = sorted(degrees.keys(),
                             key=lambda x: degrees[x], reverse=True)[:k]

    return valid_nodes[:k]


def enhanced_kmeans(G, k, max_iter=100):
    """Enhanced Advanced K-Means with hybrid initialization"""
    # Precompute shortest path distances
    distance_matrix = compute_shortest_path_distances(G)
    nodes = list(G.nodes())

    # Calculate average node degree
    degrees = dict(G.degree())
    avg_degree = np.mean(list(degrees.values()))
    rho = round(avg_degree)

    # Hybrid initialization
    controllers = enhanced_initial_controllers(G, k)
    prev_controllers = None
    clusters = {c: [] for c in controllers}

    iter_count = 0
    while iter_count < max_iter and controllers != prev_controllers:
        prev_controllers = controllers.copy()

        # Assign nodes to nearest controller
        clusters = {c: [] for c in controllers}
        for node in nodes:
            if node in controllers:
                continue
            nearest = min(controllers, key=lambda c: distance_matrix[node][c])
            clusters[nearest].append(node)

        # Update centroids with load awareness
        new_controllers = []
        for cluster in clusters.values():
            if not cluster:
                new_controllers.append(controllers[len(new_controllers)])
                continue

            # Find node with minimum total distance and sufficient degree
            cluster_nodes = [n for n in cluster if degrees[n] >= rho]  # Use degrees dict
            if not cluster_nodes:
                cluster_nodes = cluster  # Fallback

            min_total = float('inf')
            best_node = cluster_nodes[0]

            for node in cluster_nodes:
                total = sum(distance_matrix[node][n] for n in cluster)
                if total < min_total:
                    min_total = total
                    best_node = node

            new_controllers.append(best_node)

        controllers = new_controllers
        iter_count += 1

    return controllers, clusters


def run_enhanced_experiment(file_path, max_controllers=12):
    """Run experiments with enhanced initialization"""
    G = read_and_preprocess_gml(file_path)

    k_values = []
    avg_times = []
    max_times = []

    for k in range(1, max_controllers + 1):
        controllers, _ = enhanced_kmeans(G, k)
        avg, max_ = calculate_response_times(G, controllers)

        k_values.append(k)
        avg_times.append(avg)
        max_times.append(max_)
        print(f"Enhanced K={k}: Avg={avg:.2f}ms, Max={max_:.2f}ms")

    # Plotting code remains similar to previous implementation
    # ... (use same visualization as original experiment runner)
