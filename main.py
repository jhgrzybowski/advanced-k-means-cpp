from utils.experiment_utils import run_latency_experiment_compare


if __name__ == "__main__":
    # List of available GML topology files in the project
    topology_files = {
        "pionier": "topologies/PionierL3_topology.gml",
        "atmnet": "topologies/Atmnet_topology.gml",
        "os3e": "topologies/Internet2_OS3E_topology.gml",
        "geant2012": "topologies/Geant2012_topology.gml"
    }

    # run_latency_experiment(gml_file=topology_files["os3e"], clustering_fn=advanced_k_means,
    #                        algorithm_name="AdvancedKMeans_OS3E", propagation_speed_km_per_ms=204, kmax=10)

    from algorithms.advanced_k_means import advanced_k_means
    from algorithms.enhanced_k_means import enhanced_k_means

    if __name__ == "__main__":
        gml_file = topology_files['os3e']
        propagation_speed_km_per_ms = 204  # przykładowa wartość
        kmax = 10

        # Przekazujesz słownik nazw do funkcji
        clustering_fns = {
            "advanced_k_means": advanced_k_means,
            "enhanced_k_means": enhanced_k_means
        }

        # Argumenty wag dla enhanced_k_means:
        enhanced_kwargs = dict(
            w_degree=0,
            w_betweenness=10,
            w_closeness=4
        )

        run_latency_experiment_compare(
            gml_file,
            clustering_fns,
            propagation_speed_km_per_ms,
            kmax,
            enhanced_k_means_kwargs=enhanced_kwargs
        )


