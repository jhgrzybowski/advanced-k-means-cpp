from utils.experiment_utils import *
from algorithms.hdids import hdids
from algorithms.advanced_k_means import advanced_kmeans
from algorithms.enhanced_k_means import enhanced_kmeans

# # Compare HDIDS
# run_controller_experiment(
#     algorithm=hdids,
#     algorithm_name="HDIDS",
#     file_path="topologies/Internet2_OS3E_topology.gml",
#     max_controllers=12
# )
#
# # Compare Advanced K-Means
# run_controller_experiment(
#     algorithm=advanced_kmeans,
#     algorithm_name="Advanced K-Means",
#     file_path="topologies/Internet2_OS3E_topology.gml",
#     max_controllers=12
# )

algorithms_to_compare = [
    # (hdids, "HDIDS"),
    (advanced_kmeans, "Advanced K-Means"),
    (enhanced_kmeans, "Enhanced K-Means"),
]

# Run comparison experiment
run_comparison_experiment(
    algorithms=algorithms_to_compare,
    topology_name="Internet2 OS3E",
    file_path="topologies/Internet2_OS3E_topology.gml",
    min_controllers=1,
    max_controllers=12



)