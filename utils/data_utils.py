import networkx as nx


def load_gml_to_delay_graph(gml_file_path, propagation_speed_km_per_ms=200000):
    """
    Loads a GML file representing a network topology and converts it to an undirected graph G,
    where the edges are weighted by propagation delays in milliseconds (ms).
    The propagation delay for each edge is calculated as: delay_ms = distance_km / propagation_speed_km_per_ms.

    Args:
        gml_file_path (str): Path to the GML file.
        propagation_speed_km_per_ms (float): Speed of signal propagation in km/ms (default: 0.2 km/ms, which is ~200,000 km/s).
            Typical values:
                - 0.2 km/ms: optical fiber (~200,000 km/s, 2/3 speed of light)
                - 0.3 km/ms: vacuum (speed of light)

    Returns:
        G (nx.Graph): An undirected NetworkX graph where edges are weighted with propagation delays (ms).
            Each node retains its GML attributes (such as 'label', 'lon', 'lat', etc.).
            Each edge has an attribute 'delay_ms' (propagation delay in ms).
    """
    # Load the graph from GML file
    G = nx.read_gml(gml_file_path, label='id')

    # Convert to undirected graph if necessary
    if not isinstance(G, nx.Graph):
        G = nx.Graph(G)

    # For each edge, compute propagation delay (in ms) based on 'dist' (distance in km)
    for u, v, data in G.edges(data=True):
        if 'dist' not in data:
            raise ValueError(f"Edge ({u}, {v}) does not have a 'dist' attribute in the GML file.")
        distance_km = float(data['dist'])
        delay_ms = distance_km / propagation_speed_km_per_ms
        data['delay_ms'] = delay_ms
        # Optionally, for clarity, overwrite 'weight' with 'delay_ms'
        data['weight'] = delay_ms

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

def compute_shortest_path_distances(G):
    """Precompute all-pairs shortest path distances.

    Args:
        G (nx.Graph): Network graph

    Returns:
        dict: Dictionary of shortest path distances
    """
    return dict(nx.all_pairs_dijkstra_path_length(G, weight='weight'))