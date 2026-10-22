# Advanced K-Means for SDN Controller Placement

Python implementation of **Advanced K-Means** for the SDN Controller Placement Problem (CPP), with request-handling latency as the primary optimization metric.

This repository accompanies the master's thesis:

> **Optimization of location of controllers in Software Defined Networks aimed at reducing the latency of serving the demands**  
> Jakub Grzybowski · supervisor: dr hab. inż. Jacek Rak · Gdańsk University of Technology (2025)

It builds on Firas Zobary's *Advanced K-Means* approach for controller-to-switch latency ([reference PDF](docs/references/Zobary_2024_Optimizing_SDN_Controller_to_Switch_Latency.pdf)).

## Documents

| Path | Description |
|------|-------------|
| [docs/thesis/Jakub_Grzybowski_Masters_Thesis.pdf](docs/thesis/Jakub_Grzybowski_Masters_Thesis.pdf) | Final master's thesis (105 pp.) |
| [docs/presentations/Defense_Presentation_Jakub_Grzybowski.pdf](docs/presentations/Defense_Presentation_Jakub_Grzybowski.pdf) | Defense slides |
| [docs/presentations/Seminar_Presentation_1.pdf](docs/presentations/Seminar_Presentation_1.pdf) | Seminar presentation 1 |
| [docs/presentations/Seminar_Presentation_2.pdf](docs/presentations/Seminar_Presentation_2.pdf) | Seminar presentation 2 |
| [docs/references/Zobary_2024_Optimizing_SDN_Controller_to_Switch_Latency.pdf](docs/references/Zobary_2024_Optimizing_SDN_Controller_to_Switch_Latency.pdf) | Base paper by Firas Zobary (2024) |

Overleaf source: [Master thesis - document](https://www.overleaf.com/project/67f27fd67ec0f4579b08ba9d)

## Algorithms

| Module | Role |
|--------|------|
| `algorithms/advanced_k_means.py` | Advanced K-Means (latency-oriented CPP) |
| `algorithms/enhanced_k_means.py` | Enhanced K-Means++ baseline (centrality-weighted) |
| `algorithms/hdids.py` | HDIDS comparison algorithm |
| `algorithms/helpers.py` | Shared helpers |

## Project layout

```text
algorithms/     # clustering / placement algorithms
experiments/    # experiment runners
topologies/     # GML / TXT network topologies (OS3E, Pionier, GEANT, …)
utils/          # data, metrics, load, plots, results helpers
results/        # experiment JSON outputs
plots/          # generated comparison plots
load/           # controller load measurements
docs/           # thesis, presentations, references
main.py         # entry point for comparative experiments
CONST.py        # shared experiment constants (topology, kmax, …)
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install networkx matplotlib numpy
```

Adjust `CONST.py` (topology key, `kmax`, …), then:

```bash
python main.py
```

`main.py` runs latency comparisons between Advanced K-Means and Enhanced K-Means++, optional multi-seed Enhanced K-Means runs, and writes JSON results under `results/`.

## Topologies

GML files under `topologies/` include Internet2 OS3E, Pionier L3, GEANT 2012, Belnet 2003, Atmnet, CESNET, IIJ, BellSouth, and others. Keys used in `main.py` map to these files (e.g. `os3e`, `pionier`, `geant`).

## Citation

If you use this code or the thesis results, please cite the thesis PDF above and Zobary (2024).
