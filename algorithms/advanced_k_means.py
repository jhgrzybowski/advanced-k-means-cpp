import networkx as nx
import numpy as np
from utils.data_utils import *

def select_initial_controllers(G, k, distance_matrix):
    """Select initial controllers using Advanced K-Means initialization.

    Args:
        G (nx.Graph): Network graph
        k (int): Number of controllers
        distance_matrix (dict): Precomputed shortest path distances

    Returns:
        list: Initial controller nodes
    """
    nodes = list(G.nodes())

    # Calculate node degrees
    degrees = dict(G.degree())
    max_degree = max(degrees.values())
    candidates = [n for n in nodes if degrees[n] == max_degree]

    # Select first controller
    if len(candidates) == 1:
        c1 = candidates[0]
    else:
        # Choose node with minimum total distance
        c1 = min(candidates, key=lambda x: sum(distance_matrix[x].values()))

    controllers = [c1]

    # Select subsequent controllers
    for _ in range(1, k):
        max_dist = -1
        next_controller = None

        for node in nodes:
            if node in controllers:
                continue
            min_dist = min(distance_matrix[node][c] for c in controllers)
            if min_dist > max_dist:
                max_dist = min_dist
                next_controller = node

        if next_controller is not None:
            controllers.append(next_controller)

    return controllers


def advanced_kmeans(G, k, max_iter=100):
    """Advanced K-Means algorithm implementation.

    Args:
        G (nx.Graph): Network graph
        k (int): Number of controllers
        max_iter (int): Maximum iterations

    Returns:
        tuple: (controllers, clusters)
    """
    distance_matrix = compute_shortest_path_distances(G)
    nodes = list(G.nodes())
    degrees = dict(G.degree())
    avg_degree = round(sum(degrees.values()) / len(nodes))

    # Initial controller selection
    controllers = select_initial_controllers(G, k, distance_matrix)
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

        # Update centroids with degree constraint
        new_controllers = []
        for cluster in clusters.values():
            if not cluster:
                new_controllers.append(controllers[len(new_controllers)])
                continue

            # Filter nodes with degree >= avg_degree
            valid_nodes = [n for n in cluster if degrees[n] >= avg_degree]
            if not valid_nodes:
                valid_nodes = cluster  # Fallback

            min_total = float('inf')
            best_node = valid_nodes[0]

            for node in valid_nodes:
                total = sum(distance_matrix[node][n] for n in cluster)
                if total < min_total:
                    min_total = total
                    best_node = node

            new_controllers.append(best_node)

        controllers = new_controllers
        iter_count += 1

    return controllers, clusters