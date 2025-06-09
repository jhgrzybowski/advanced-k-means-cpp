import networkx as nx

def _compute_path_lengths(G):
    """
    Returns paths computed for the given network using Dijkstra's algorithm.
    Args:
        G (nx.Graph): Input graph.

    Returns:
        path_lengths (dict): Shortest delays path.
    """
    return dict(nx.all_pairs_dijkstra_path_length(G, weight='delay_ms'))

def _compute_nodes_degrees(G):
    """
    Computes degree of each node in the given network
     Args:
        G (nx.Graph): Input graph.

    Returns:
        degrees (dict): Degree of each node.
    """

    return dict(G.degree())


def _compute_node_average_degree(degrees):
    """
    Computes the average degree of nodes in the graph, rounded to the nearest integer.

    Args:
        degrees (dict): Dictionary of degree of each node.

    Returns:
        avg_degree (int): Average degree of nodes in the graph (rounded).
    """

    # Sum up degrees in the network
    total_degree_count = sum(degrees.values())

    # Count up the nodes
    num_nodes = len(degrees)

    return int(round(total_degree_count / num_nodes)) if num_nodes else 0

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

def _prepare_graph_features(G):
    """
    Prepares features required for clustering/controller placement algorithms:
    - degree for each node
    - shortest path lengths
    - eligible nodes for initial center selection

    Args:
        G (nx.Graph): Input graph (should have delays on edges if required).

    Returns:
        eligible_nodes (list): Nodes eligible for being an initial center.
        degrees (dict): Node degrees {node: degree}.
        path_lengths (dict): Shortest path lengths {source: {target: distance}}.
    """

    degrees = dict(G.degree())

    # Node average degree | Equation (13)
    avg_degree = _compute_node_average_degree(degrees)

    # Select eligible nodes to be selected as initial center | Equation (14)
    eligible_nodes = _eligible_center_nodes(degrees, avg_degree)
    return eligible_nodes, degrees

def _prepare_weight_normalization(G, n, betweenness, closeness):

    eligible_nodes, degrees = _prepare_graph_features(G)

    deg_values = [degrees[n] for n in eligible_nodes]
    betw_values = [betweenness[n] for n in eligible_nodes]
    close_values = [closeness[n] for n in eligible_nodes]

    def norm(val, vmin, vmax):
        return 0 if vmax == vmin else (val - vmin) / (vmax - vmin)

    deg_min, deg_max = min(deg_values), max(deg_values)
    betw_min, betw_max = min(betw_values), max(betw_values)
    close_min, close_max = min(close_values), max(close_values)

    d_norm = norm(degrees[n], deg_min, deg_max)
    b_norm = norm(betweenness[n], betw_min, betw_max)
    c_norm = norm(closeness[n], close_min, close_max)

    return d_norm, b_norm, c_norm


