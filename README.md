# GDAR: Graph Diffusion Autoregressive Model

[![Documentation Status](https://readthedocs.org/projects/gdar/badge/?version=latest)](https://gdar.readthedocs.io/en/latest/?badge=latest)

---

## Overview

This library implements the **Graph Diffusion Autoregressive (GDAR) model**. 
The framework combines classical Vector Autoregression (VAR) with a graph diffusion processes,
leveraging structural connectivity priors to estimate **directed, time-resolved flow signals** in neural data.

For more details, see our [paper](https://www.biorxiv.org/content/10.1101/2024.02.26.582177v2).

---

## Installation

Install the latest release from PyPI:

```bash
git clone https://github.com/yourusername/gdar.git
cd gdar
pip install -e .
```
## Basic Usage

```python
from gdar.graph import Graph
from gdar.gdar_model import GDARModel
import numpy as np

# Create a simple graph from an edge list
edges = [(0, 1), (1, 2), (2, 3)]
graph = Graph()
graph.generate_from_edge_list(edges)

# Generate synthetic data
N, T = 4, 500
data = np.random.randn(N, T)

# Initialize and fit GDAR model
model = GDARModel(graph=graph, K=5)
coefficients = model.fit_gdar(data)

print("GDAR coefficients shape:", coefficients.shape)
```

## Documentation

Full documentation is available at: https://gdar.readthedocs.io

It includes:
- Getting Started guide
- API Reference
- Tutorials and advanced usage

## Citation
If you use GDAR in your research, please cite:
```css
Schwock, F., Bloch, J., Khateeb, K., Zhou, J., Atlas, L., & Yazdan-Shahmorad, A.
"Inferring Neural Communication Dynamics from Field Potentials Using Graph Diffusion Autoregression."
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.