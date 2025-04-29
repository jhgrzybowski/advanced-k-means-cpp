from experiment_utils import run_controller_experiment
from hdids import hdids_algorithm
from advanced_k_means import advanced_kmeans

# Compare HDIDS
run_controller_experiment(
    algorithm=hdids_algorithm,
    algorithm_name="HDIDS",
    file_path="Geant2012_topology.gml",
    max_controllers=12
)

# Compare Advanced K-Means
run_controller_experiment(
    algorithm=advanced_kmeans,
    algorithm_name="Advanced K-Means",
    file_path="Geant2012_topology.gml",
    max_controllers=12
)