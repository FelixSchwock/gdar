Usage Examples
==============

This section provides examples of how to use the `gdar` package for creating graphs,
visualizing connectivity, and fitting the Graph Diffusion Autoregressive (GDAR) model.
The GitHub repository also contains a demo jupyter notebook that demonstrates these and additional functionalities in
more detail.

.. contents::
   :local:
   :depth: 2

Creating a Structural Connectivity Graph
----------------------------------------

The `gdar` package provides functions for creating structural connectivity graphs, which
the GDAR model uses to estimate network communication dynamics. You can create graphs from:

- A user-defined edge list
- Node positions and built-in graph generator methods

First, let's create a graph from a simple edge list:

.. code-block:: python

    from gdar.graph import Graph

    # Define edges
    edge_list = [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (3, 4)]

    # Create graph
    graph = Graph()
    graph.generate_from_edge_list(edge_list)

Visualizing the Graph
----------------------

To visualize the graph, use the `plot_graph` function from the `visualization` module:

.. code-block:: python

    from gdar.visualization import plot_graph
    from matplotlib import pyplot as plt

    node_positions = {
        0: (0, 0),
        1: (1, 0),
        2: (1, 1),
        3: (0, 1),
        4: (0.5, 1.5)
    }

    graph.set_node_position_name(node_positions)
    plot_graph(
        graph,
        directed=False,
        node_size=500,
        node_color='lightblue',
        edge_color='gray',
        width=10,
        with_labels=True
    )
    plt.show()

Adding Edges to an Existing Graph
---------------------------------

You can add edges manually:

.. code-block:: python

    edges_to_add = [(0, 3), (2, 4)]
    graph.add_edges(edges_to_add)

    # Visualize updated graph
    plot_graph(graph, directed=False, node_size=500, node_color='lightblue', edge_color='gray')
    plt.show()

Creating a Graph from Node Positions
------------------------------------

For large arrays or multiple brain regions, you can use graph generator methods:

.. code-block:: python

    import pickle
    from gdar.graph import Graph

    # Load node positions
    with open('demo_files/node_positions.pkl', 'rb') as f:
        node_positions = pickle.load(f)

    # Create proximity-based graph
    graph2 = Graph()
    graph2.proximity_graph(node_positions, dist_th=2)

    # Visualize
    plot_graph(graph2, directed=False, node_size=500, node_color='lightblue')
    plt.show()

Fitting the GDAR Model
-----------------------

Once a graph is created, fit the GDAR model to your time-series data:

.. code-block:: python

    import numpy as np
    from gdar.gdar_model import GDARModel

    # Generate toy data
    np.random.seed(42)
    T = 1000  # number of time points
    N = len(node_positions)
    data = np.random.randn(N, T)

    # Initialize GDAR model
    model_order = 5
    gdar_model = GDARModel(graph=graph2, K=model_order)

    # Fit model
    coefficients = gdar_model.fit_gdar(data)
    print(f'Fitted coefficients shape: {coefficients.shape}')


