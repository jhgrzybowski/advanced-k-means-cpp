import os
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

plt.rcParams['font.family'] = 'Arial'

# --- Helper function for all plots ---
def _make_plot(
    x,
    y_data,
    y_labels,
    y_err=None,
    colors=None,
    markers=None,
    xlabel="",
    ylabel="",
    title="",
    legend_loc="best",
    ylim=None,
    ytick_major=None,
    fname="plot.png",
    show=True,
    grid_style=":",
    tight=True,
):
    """
    Generic helper to plot lines or errorbars for multiple series.
    """
    plt.figure(figsize=(12,6))
    n = len(y_data)
    for i in range(n):
        color = colors[i] if colors else None
        marker = markers[i] if markers else None
        label = y_labels[i]
        if y_err is not None and y_err[i] is not None:
            plt.errorbar(
                x, y_data[i], yerr=y_err[i],
                fmt=marker if marker else '-', color=color,
                capsize=4, label=label
            )
        else:
            plt.plot(x, y_data[i], marker=marker, color=color, label=label)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(loc=legend_loc)
    plt.grid(True, linestyle=grid_style)

    ax = plt.gca()
    if ytick_major:
        ax.yaxis.set_major_locator(MultipleLocator(ytick_major))
    ax.xaxis.set_major_locator(MultipleLocator(1))
    if ylim:
        plt.ylim(ylim)
    if tight:
        plt.tight_layout()
    plt.savefig(fname, bbox_inches='tight', dpi=150)
    print(f"Plot saved: {fname}")
    if show:
        plt.show()
    plt.close()

# --- First experiment: Advanced vs Enhanced K-Means (means only) ---
def plot_latency_comparison(
    k_values,
    avg_latencies,
    max_latencies,
    clustering_fns,
    experiment_name,
    topology_name,
    output_dir="plots"
):
    """
    Plot mean avg/max latency for advanced & enhanced k-means, using unified styling.
    """
    os.makedirs(output_dir, exist_ok=True)

    color_map = {
        'advanced_k_means': '#003366',    # Pantone 540C
        'enhanced_k_means': '#C8102E'     # Pantone 1797C
    }
    marker_map = {
        'advanced_k_means': 'o',
        'enhanced_k_means': 's'
    }
    names = list(clustering_fns.keys())
    colors = [color_map.get(n, None) for n in names]
    markers = [marker_map.get(n, None) for n in names]

    info_str = f"{experiment_name}__{topology_name}"

    # --- Average Latency ---
    _make_plot(
        x=k_values,
        y_data=[avg_latencies[n] for n in names],
        y_labels=[n.replace('_', ' ').title() for n in names],
        colors=colors,
        markers=markers,
        xlabel='Number of Controllers (K)',
        ylabel='Average Response Time [ms]',
        title=f'Average Latency – {experiment_name} – Topology: {topology_name}',
        legend_loc="best",
        ylim=(0, 8),
        ytick_major=0.5,
        fname=f"{output_dir}/1_{info_str} (150dpi).png"
    )

    # --- Max Latency ---
    _make_plot(
        x=k_values,
        y_data=[max_latencies[n] for n in names],
        y_labels=[n.replace('_', ' ').title() for n in names],
        colors=colors,
        markers=markers,
        xlabel='Number of Controllers (K)',
        ylabel='Maximum Response Time [ms]',
        title=f'Maximum Latency – {experiment_name} – Topology: {topology_name}',
        legend_loc="best",
        ylim=(0, 16),
        ytick_major=0.5,
        fname=f"{output_dir}/2_{info_str} (150dpi).png"
    )

# --- Second experiment: Enhanced K-Means (mean ± std) and Advanced K-Means ---
def plot_enhanced_kmeans_experiment(
    k_values,
    avg_delays_advanced,
    max_delays_advanced,
    avg_delays_enhanced,
    std_avg_delays_enhanced,
    max_delays_enhanced,
    std_max_delays_enhanced,
    experiment_name,
    topology_name,
    output_dir="plots"
):
    """
    Plot average (with std) and maximum (with std) delay for enhanced k-means++ versus advanced k-means.
    """
    os.makedirs(output_dir, exist_ok=True)
    info_str = f"{experiment_name}__{topology_name}"

    # --- Average Delay Plot (mean ± std) ---
    _make_plot(
        x=k_values,
        y_data=[avg_delays_advanced, avg_delays_enhanced],
        y_labels=[
            "Advanced K-Means",
            "Enhanced K-Means++ (mean ± std)"
        ],
        y_err=[None, std_avg_delays_enhanced],
        colors=["#003366", "#C8102E"],
        markers=["o", "s"],
        xlabel="Number of Controllers (K)",
        ylabel="Average Response Time [ms]",
        title=f"Average Propagation Delay – {experiment_name} – Topology: {topology_name}",
        legend_loc="best",
        ylim=(0, 8),
        ytick_major=0.5,
        fname=f"{output_dir}/3_{info_str} (150dpi).png"
    )

    # --- Max Delay Plot (mean ± std) ---
    _make_plot(
        x=k_values,
        y_data=[max_delays_advanced, max_delays_enhanced],
        y_labels=[
            "Advanced K-Means",
            "Enhanced K-Means++ (mean ± std)"
        ],
        y_err=[None, std_max_delays_enhanced],
        colors=["#003366", "#C8102E"],
        markers=["o", "s"],
        xlabel="Number of Controllers (K)",
        ylabel="Maximum Response Time [ms]",
        title=f"Maximum Propagation Delay – {experiment_name} – Topology: {topology_name}",
        legend_loc="best",
        ylim=(0, 16),
        ytick_major=0.5,
        fname=f"{output_dir}/4_{info_str} (150dpi).png"
    )
