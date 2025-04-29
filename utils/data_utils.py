import math
import networkx as nx

def read_and_preprocess_gml(file_path):
    """
    Reads a .gml file and preprocesses the graph by calculating edge weights based on propagation delay.

    Parameters:
        file_path (str): Path to the .gml file.

    Returns:
        nx.Graph: Preprocessed undirected graph with edge weights as propagation delay (ms).
    """
    G = nx.read_gml(file_path)
    G = G.to_undirected()

    for u, v in G.edges():
        u_data = G.nodes[u]
        v_data = G.nodes[v]

        # Extract coordinates using 'lon'/'lat' keys
        lon1 = u_data.get('lon', 0.0)
        lat1 = u_data.get('lat', 0.0)
        lon2 = v_data.get('lon', 0.0)
        lat2 = v_data.get('lat', 0.0)

        # Haversine formula
        R = 6371000  # Earth radius in meters
        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        delta_phi = math.radians(lat2 - lat1)
        delta_lambda = math.radians(lon2 - lon1)

        a = (math.sin(delta_phi / 2.0) ** 2 +
             math.cos(phi1) * math.cos(phi2) *
             math.sin(delta_lambda / 2.0) ** 2)

        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        distance = R * c

        # Propagation delay in milliseconds
        G[u][v]['weight'] = (distance / 3e8) * 1000  # Speed of light

    return G


def haversine_heuristic(u, v, G):
    """
    Heuristic function for A* algorithm using Haversine distance between nodes.

    Parameters:
        u, v (nodes): Nodes to calculate heuristic between.
        G (nx.Graph): Network graph.

    Returns:
        float: Estimated propagation delay (ms) between u and v.
    """
    u_data = G.nodes[u]
    v_data = G.nodes[v]

    lon1 = u_data.get('lon', 0.0)
    lat1 = u_data.get('lat', 0.0)
    lon2 = v_data.get('lon', 0.0)
    lat2 = v_data.get('lat', 0.0)

    R = 6371000
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = (math.sin(delta_phi / 2.0) ** 2 +
         math.cos(phi1) * math.cos(phi2) *
         math.sin(delta_lambda / 2.0) ** 2)

    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return (R * c / 3e8) * 1000  # ms