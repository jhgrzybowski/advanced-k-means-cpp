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
                           algorithm_name="AdvancedKMeans_OS3E", kmax=12)

