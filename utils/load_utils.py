import os
import json
import random
from utils.data_utils import load_gml_to_delay_graph

from CONST import *

dir_path=f"load/{topo_dir}"

def compute_controller_load(clusters):
    """
    Computes the load for each controller as the number of switches assigned to it.

    Args:
        clusters (dict): Mapping {controller_id: set or list of assigned node ids}.

    Returns:
        dict: {
            'controller_loads': dict {controller_id: int},
            'max_controller_load': int
        }
    """
    controller_loads = {str(int(ctrl)): len(members) for ctrl, members in clusters.items()}
    max_controller_load = max(controller_loads.values()) if controller_loads else 0
    return {
        'controller_loads': controller_loads,
        'max_controller_load': max_controller_load
    }

def run_and_save_controller_loads(
    gml_file,
    propagation_speed_km_per_ms,
    k_max,
    clustering_fns,
    seed,
    enhanced_k_means_kwargs=None
):
    """
    For each k in 1...k_max, runs Advanced K-Means and Enhanced K-Means (single run each),
    computes controller loads, and saves results as JSON in "load/" directory.
    All k results are written as a list to 'advanced_k-means_load.json' and 'enhanced_k-means_load.json'.

    Args:
        gml_file (str): Path to the network topology in GML format.
        propagation_speed_km_per_ms (float): Propagation speed for the delay calculation.
        k_max (int): Maximum number of controllers/clusters (runs for k=1...k_max).
        clustering_fns (dict): Dict {"advanced_k_means": fn, "enhanced_k_means": fn}.
        seed (int): Random seed for reproducibility.
        enhanced_k_means_kwargs (dict): Keyword arguments for enhanced_k_means (weights etc.).

    Returns:
        dict: {
            'advanced_k_means': [result for each k],
            'enhanced_k_means': [result for each k]
        }
    """
    os.makedirs(dir_path, exist_ok=True)
    from copy import deepcopy
    G_orig = load_gml_to_delay_graph(gml_file, propagation_speed_km_per_ms)
    rng = random.Random(seed)
    kwargs = enhanced_k_means_kwargs or {}

    txt_content_adv = "======= ADVANCED K-MEANS ======\n"
    txt_content_enh = "======= ENHANCED K-MEANS ======\n"

    advanced_results = []
    enhanced_results = []

    for k in range(1, k_max + 1):
        # Use deepcopy to keep G pristine for both algorithms (in case of in-place changes)
        G = deepcopy(G_orig)

        # --- Advanced K-Means ---
        adv_controllers, adv_clusters = clustering_fns["advanced_k_means"](G, k)
        adv_load = compute_controller_load(adv_clusters)
        adv_result = {
            "k": k,
            "controllers": list(map(int, adv_controllers)),
            "controller_loads": adv_load["controller_loads"],
            "max_controller_load": adv_load["max_controller_load"],
            "clusters": {str(int(c)): list(map(int, members)) for c, members in adv_clusters.items()}
        }
        advanced_results.append(adv_result)

        txt_content_adv += f"K = {k}: Max load = {adv_load['max_controller_load']}\n"

        # --- Enhanced K-Means ---
        enh_controllers, enh_clusters = clustering_fns["enhanced_k_means"](G, k, rng, **kwargs)
        enh_load = compute_controller_load(enh_clusters)
        enh_result = {
            "k": k,
            "controllers": list(map(int, enh_controllers)),
            "controller_loads": enh_load["controller_loads"],
            "max_controller_load": enh_load["max_controller_load"],
            "clusters": {str(int(c)): list(map(int, members)) for c, members in enh_clusters.items()}
        }
        enhanced_results.append(enh_result)

        txt_content_enh += f"K = {k}: Max load = {enh_load['max_controller_load']}\n"

    # Save all advanced results in one file
    with open(f"{dir_path}/advanced_k-means_load.json", "w") as f:
        json.dump(advanced_results, f, indent=2)
    print("Advanced K-Means loads saved to load/advanced_k-means_load.json")

    # Save all enhanced results in one file
    with open(f"{dir_path}/enhanced_k-means_load.json", "w") as f:
        json.dump(enhanced_results, f, indent=2)
    print("Enhanced K-Means loads saved to load/enhanced_k-means_load.json")

    with open(f"{dir_path}/advanced_max_load.txt", "w") as f:
        f.write(txt_content_adv)
    with open(f"{dir_path}/enhanced_max_load.txt", "w") as f:
        f.write(txt_content_enh)

    return {
        "advanced_k_means": advanced_results,
        "enhanced_k_means": enhanced_results
    }
