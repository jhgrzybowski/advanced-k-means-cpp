

from utils.experiment_utils import run_latency_experiment
from algorithms.advanced_k_means import advanced_k_means

if __name__ == "__main__":
    # List of available GML topology files in the project
    topology_files = {
        "pionier": "topologies/PionierL3_topology.gml",
        "atmnet": "topologies/Atmnet_topology.gml",
        "os3e": "topologies/Internet2_OS3E_topology.gml",
        "geant2012": "topologies/Geant2012_topology.gml"
    }

    run_latency_experiment(gml_file=topology_files["os3e"], clustering_fn=advanced_k_means,
                           algorithm_name="AdvancedKMeans_OS3E", propagation_speed_km_per_ms=204, kmax=10)

    # Załaduj topologię


    # from utils.data_utils import load_gml_to_delay_graph
    # import networkx as nx
    #
    # G = load_gml_to_delay_graph("topologies/PionierL3_topology.gml")
    #
    # # Wybierz przykładowe pary (tu: ID węzłów, zmień na te, które istnieją w grafie)
    # pairs = [
    #     (0, 34),  # Warszawa - Poznań
    #     (30, 36),  # Gdańsk - Toruń
    #     (8, 19)  # Rzeszów - Zamość
    # ]
    #
    # for u, v in pairs:
    #     # Najkrótsza ścieżka wg opóźnienia propagacyjnego (w ms)
    #     try:
    #         delay = nx.shortest_path_length(G, source=u, target=v, weight='delay_ms')
    #         print(
    #             f"Propagation delay between {G.nodes[u]['label']} ({u}) and {G.nodes[v]['label']} ({v}): {delay:.4f} ms")
    #     except Exception as e:
    #         print(f"Cannot compute delay between {u} and {v}: {e}")

