from utils.load_utils import run_and_save_controller_loads
from utils.results_utils import save_results_to_json
from experiments.experiments_runner import run_enhanced_kmeans_experiment
from experiments.experiments_runner import run_latency_experiment_compare

from CONST import *

if __name__ == "__main__":
    # List of available GML topology files in the project
    topology_files = {
        "pionier": "topologies/PionierL3_topology.gml",
        "atmnet": "topologies/Atmnet_topology.gml",
        "os3e": "topologies/Internet2_OS3E_topology.gml",
        "geant": "topologies/Geant2012_topology.gml",
        "abvt": "topologies/Abvt.gml",
        "gts": "topologies/GtsSlovakia.gml",
        "iij": "topologies/Iij.gml",
        "hurricane": "topologies/HurricaneElectric.gml",
        "jpn": "topologies/WideJpn.gml",
        "bell": "topologies/Bellsouth.gml",
        "belnet": "topologies/Belnet2003.gml",
        "cesnet": "topologies/Cesnet1999.gml",
    }

    # run_latency_experiment(gml_file=topology_files["os3e"], clustering_fn=advanced_k_means,
    #                        algorithm_name="AdvancedKMeans_OS3E", propagation_speed_km_per_ms=204, kmax=10)

    from algorithms.advanced_k_means import advanced_k_means
    from algorithms.enhanced_k_means import enhanced_k_means

    if __name__ == "__main__":

        # Experiments paremeters
        gml_file = topology_files[f'{topo_dir}']
        propagation_speed_km_per_ms = 204
        seed = 42
        enhanced_algorithm_runs = 10
        k_value = range(1,kmax+1)

        clustering_fns = {
            "advanced_k_means": advanced_k_means,
            "enhanced_k_means": enhanced_k_means
        }

        # Enhanced K-Means kwargs
        enhanced_kwargs = dict(
            w_degree=0.2,
            w_betweenness=0.4,
            w_closeness=0.4
        )

        run_latency_experiment_compare(
            gml_file,
            clustering_fns,
            propagation_speed_km_per_ms,
            kmax,
            seed,
            enhanced_kwargs,
            k_value
        )

        run_enhanced_kmeans_experiment(
            gml_file,
            clustering_fns,
            propagation_speed_km_per_ms,
            kmax,
            enhanced_algorithm_runs,
            seed,
            enhanced_kwargs
        )

        save_results_to_json(
            gml_file,
            clustering_fns,
            propagation_speed_km_per_ms,
            kmax,
            enhanced_algorithm_runs,
            seed,
            enhanced_kwargs
        )

        run_and_save_controller_loads(
            gml_file,
            propagation_speed_km_per_ms,
            kmax,
            clustering_fns,
            seed,
            enhanced_kwargs
        )



