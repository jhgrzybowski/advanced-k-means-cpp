import networkx as nx


def compute_shortest_paths(graph):
    """
    Compute shortest path distances between all pairs of nodes.

    Parameters:
        graph (nx.Graph): Network topology.

    Returns:
        dict: Shortest path distances between all node pairs.
    """
    return dict(nx.all_pairs_shortest_path_length(graph))


def calculate_node_degrees(graph):
    """
    Calculate the degree (number of connections) for each node.

    Parameters:
        graph (nx.Graph): Network topology.

    Returns:
        dict: Node degrees.
    """
    return dict(graph.degree())


def compute_average_degree(node_degrees):
    """
    Compute the average node degree (rounded to nearest integer).

    Parameters:
        node_degrees (dict): Degrees of all nodes.

    Returns:
        int: Average node degree.
    """
    total = sum(node_degrees.values())
    n = len(node_degrees)
    return round(total / n)


def select_first_center(graph, node_degrees, shortest_paths, rho):
    """
    Select the first controller center based on highest degree and minimum sum of shortest paths.

    Parameters:
        graph (nx.Graph): Network topology.
        node_degrees (dict): Node degrees.
        shortest_paths (dict): Shortest path distances.
        rho (int): Average node degree.

    Returns:
        object: First controller node.
    """
    candidates = [node for node, degree in node_degrees.items() if degree >= rho]
    if not candidates:
        raise ValueError("No nodes with degree >= average degree.")

    sum_distances = {node: sum(shortest_paths[node].values()) for node in candidates}
    return min(candidates, key=lambda x: sum_distances[x])


def select_next_centers(graph, existing_centers, shortest_paths, rho, node_degrees, k):
    """
    Select subsequent controller centers as the farthest nodes from existing centers.

    Parameters:
        graph (nx.Graph): Network topology.
        existing_centers (list): Current controller centers.
        shortest_paths (dict): Shortest path distances.
        rho (int): Average node degree.
        node_degrees (dict): Node degrees.
        k (int): Total number of controllers required.

    Returns:
        list: New controller centers.
    """
    centers = existing_centers.copy()
    while len(centers) < k:
        candidates = [
            node for node in graph.nodes()
            if node not in centers and node_degrees.get(node, 0) >= rho
        ]
        if not candidates:
            candidates = [node for node in graph.nodes() if node not in centers]

        max_distance = -1
        next_center = None
        for node in candidates:
            min_dist = min([shortest_paths[node][c] for c in centers], default=float('inf'))
            if min_dist > max_distance or (
                    min_dist == max_distance and node_degrees[node] > node_degrees.get(next_center, 0)):
                max_distance = min_dist
                next_center = node

        if next_center is None:
            break
        centers.append(next_center)
    return centers[len(existing_centers):]


def assign_clusters(centers, shortest_paths):
    """
    Assign each node to the nearest controller center.

    Parameters:
        centers (list): Controller centers.
        shortest_paths (dict): Shortest path distances.

    Returns:
        dict: Clusters with nodes grouped under each center.
    """
    clusters = {center: [] for center in centers}
    for node in shortest_paths:
        closest = min(centers, key=lambda c: shortest_paths[node][c])
        clusters[closest].append(node)
    return clusters


def update_centroids(clusters, shortest_paths, node_degrees, rho):
    """
    Update centroids within each cluster to minimize intra-cluster distances.

    Parameters:
        clusters (dict): Current clusters.
        shortest_paths (dict): Shortest path distances.
        node_degrees (dict): Node degrees.
        rho (int): Average node degree.

    Returns:
        list: Updated controller centers.
    """
    new_centers = []
    for cluster in clusters.values():
        candidates = [n for n in cluster if node_degrees.get(n, 0) >= rho]
        if not candidates:
            candidates = cluster

        sum_dist = {n: sum(shortest_paths[n][m] for m in cluster) for n in candidates}
        new_center = min(sum_dist, key=lambda x: sum_dist[x])
        new_centers.append(new_center)
    return new_centers


def advanced_kmeans(graph, K):
    """
    Advanced K-Means algorithm for SDN Controller Placement.

    Parameters:
        graph (nx.Graph): Network topology.
        K (int): Number of controllers.

    Returns:
        list: Optimal controller positions.
    """
    shortest_paths = compute_shortest_paths(graph)
    node_degrees = calculate_node_degrees(graph)
    rho = compute_average_degree(node_degrees)

    # Select initial centers
    first_center = select_first_center(graph, node_degrees, shortest_paths, rho)
    centers = [first_center]

    if K > 1:
        centers += select_next_centers(graph, centers, shortest_paths, rho, node_degrees, K)

    prev_centers = None
    max_iterations = 100
    iteration = 0

    while centers != prev_centers and iteration < max_iterations:
        prev_centers = centers.copy()
        clusters = assign_clusters(centers, shortest_paths)
        centers = update_centroids(clusters, shortest_paths, node_degrees, rho)
        iteration += 1

    return centers