from utils.experiment_utils import run_latency_experiment
from algorithms.advanced_k_means import advanced_k_means

if __name__ == "__main__":
    # List of available GML topology files in the project
    topology_files = [
        "topologies/PionierL3_topology.gml",
        "topologies/Atmnet_topology.gml",
        "topologies/Internet2_OS3E_topology.gml",
        "topologies/Geant2012_topology.gml"
    ]

    run_latency_experiment(gml_file=topology_files[2], clustering_fn=advanced_k_means,
                           algorithm_name="AdvancedKMeans_OS3E", kmax=12)

    # k = 5  # Example: number of controllers for all topologies
    # for file in topology_files:
    #     print("=" * 60)
    #     print(f"Topology: {file}")
    #     try:
    #         G = load_gml_to_delay_graph(file)
    #         controllers, clusters = advanced_k_means(G, k=k)
    #         print("Controllers (IDs):", controllers)
    #         for ctrl in controllers:
    #             ctrl_label = G.nodes[ctrl].get('label', str(ctrl))
    #             members_labels = [G.nodes[n].get('label', str(n)) for n in clusters[ctrl]]
    #             print(f"  Controller {ctrl} ({ctrl_label}): {members_labels}")
    #     except Exception as e:
    #         print(f"  Error processing {file}: {e}")
    # print("=" * 60)

