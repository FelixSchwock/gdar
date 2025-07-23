from matplotlib import pyplot as plt
import numpy as np
import networkx as nx
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
	"""
	Truncate a colormap to a specified range.

	Parameters:
		cmap (Colormap): The colormap to truncate.
		minval (float): The lower bound of the range to keep (0.0 to 1.0).
		maxval (float): The upper bound of the range to keep (0.0 to 1.0).
		n (int): Number of colors in the truncated colormap.
	Returns:
		LinearSegmentedColormap: A new colormap that is truncated to the specified range.
	"""
	new_cmap = LinearSegmentedColormap.from_list(
		'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
		cmap(np.linspace(minval, maxval, n)))
	return new_cmap

def set_edge_weights(edge_list, edge_weights=None):
	"""
	Set weights for edges in a graph. This is needed for creating a NetworkX graph with weighted edges.

	Parameters:
		edge_list (list): List of edges in the graph, where each edge is a tuple (node1, node2).
		edge_weights (list, optional): List of weights corresponding to each edge. If None, all edges are assigned a
			weight of 1.
	Returns:
		list: List of edges with weights, where each edge is a tuple (node1, node2, {'weight': weight}).
	"""
	if edge_weights is None:
		edge_weights = np.ones(len(edge_list))
	elif len(edge_weights) != len(edge_list):
		raise TypeError('edge signal s1 must have same length as edge_list')

	edge_list_with_weights = []
	for i in range(len(edge_weights)):
		if edge_weights[i] < 0:
			# flip direction of edge if weight is negative
			edge_list_with_weights.append((edge_list[i][1], edge_list[i][0], {'weight': -edge_weights[i]}))
		else:
			edge_list_with_weights.append((edge_list[i][0], edge_list[i][1], {'weight': edge_weights[i]}))

	return edge_list_with_weights

def plot_graph(graph, directed=True, **kwargs):
	"""
	Plot graph using NetworkX.

	Parameters:
		graph (:class:`gdar.graph.Graph`): Graph object containing edge_list and node_positions.
		directed (bool): If True, plot as directed graph; otherwise, as undirected.
		kwargs: Additional keyword arguments for the NetworkX draw function.
	"""

	edge_list_with_weights = set_edge_weights(graph.edge_list, edge_weights=None)

	if not directed:
		G = nx.Graph(edge_list_with_weights)
	else:
		G = nx.DiGraph(edge_list_with_weights)

	if 'width' not in kwargs:
		kwargs['width'] = 3.0
	if 'node_size' not in kwargs:
		kwargs['node_size'] = 100
	if 'node_color' not in kwargs:
		kwargs['node_color'] = 'lightgray'
	elif type(kwargs['node_color']) == list:
		# reorder node_color according to node_list in G
		kwargs['node_color'] = [kwargs['node_color'][node] for node in G]

	nx.draw(G, pos=graph.node_positions, **kwargs)

def plot_flow_graph(graph, flow_vector, directed=True, **kwargs):
	"""
	Plot flow graph using NetworkX.

	Parameters:
		graph (:class:`gdar.graph.Graph`): Graph object containing edge_list and node_positions.
		flow_vector (np.ndarray): Flow vector with flow values for each edge.
		directed (bool): If True, plot as directed graph; otherwise, as undirected.
		kwargs: Additional keyword arguments for the NetworkX draw function. The following keyword arguments are set
		automatically if not provided, or are in addition to the ones already set by NetworkX draw function:
			- edge_cmap: Colormap for edge weights (default: cm.bwr).
			- vlim_quantile: Quantile for setting vmin and vmax of the colormap (default: None).
			- edge_vmin: Minimum value for edge weights (default: min(flow_vector)).
			- edge_vmax: Maximum value for edge weights (default: max(flow_vector)).
			- width: Width of edges (default: 3.0).
			- node_size: Size of nodes (default: 100).
			- node_color: Color of nodes (default: 'lightgray').
			- color_label: Label for the colorbar (default: 'flow magnitude').
			- extend: Extend option for the colorbar (default: 'neither').
			- show_colormap: Whether to show the colormap (default: True).
			- cmap_pad: Padding for the colormap (default: -0.05).
			- edgelist: List of edges to highlight in the plot.
			- ax: Matplotlib axis to draw on (default: current axis).
	"""
	edge_list_with_weights = set_edge_weights(graph.edge_list, edge_weights=flow_vector)

	if not directed:
		G = nx.Graph(edge_list_with_weights)
	else:
		G = nx.DiGraph(edge_list_with_weights)

	kwargs_extended = {}

	if 'edge_cmap' not in kwargs:
		kwargs['edge_cmap'] = cm.bwr
	if 'vlim_quantile' in kwargs:
		vlim = np.quantile(np.abs(flow_vector), kwargs['vlim_quantile'])
		kwargs['edge_vmin'] = -vlim
		kwargs['edge_vmax'] = vlim
		del kwargs['vlim_quantile']
	if 'edge_vmin' not in kwargs:
		kwargs['edge_vmin'] = np.min(flow_vector)
	if 'edge_vmax' not in kwargs:
		kwargs['edge_vmax'] = np.max(flow_vector)
	if 'width' not in kwargs:
		kwargs['width'] = 3.0
	if 'node_size' not in kwargs:
		kwargs['node_size'] = 100
	if 'node_color' not in kwargs:
		kwargs['node_color'] = 'lightgray'
	elif type(kwargs['node_color']) == list:
		# reorder node_color according to node_list in G
		kwargs['node_color'] = [kwargs['node_color'][node] for node in G]
	if 'color_label' not in kwargs:
		kwargs_extended['color_label'] = 'flow magnitude'
	else:
		kwargs_extended['color_label'] = kwargs['color_label']
		del kwargs['color_label']
	if 'extend' not in kwargs:
		kwargs_extended['extend'] = 'neither'
	else:
		kwargs_extended['extend'] = kwargs['extend']
		del kwargs['extend']
	if 'show_colormap' not in kwargs:
		kwargs_extended['show_colormap'] = True
	else:
		kwargs_extended['show_colormap'] = kwargs['show_colormap']
		del kwargs['show_colormap']
	if 'cmap_pad' not in kwargs:
		kwargs_extended['cmap_pad'] = -0.05
	else:
		kwargs_extended['cmap_pad'] = kwargs['cmap_pad']
		del kwargs['cmap_pad']
	if 'edgelist' in kwargs:
		edge_list_index = []
		edge_list_full = list(G.edges())
		for e in kwargs['edgelist']:
			if e in edge_list_full:
				edge_list_index.append(edge_list_full.index(e))
			elif e[::-1] in edge_list_full:
				edge_list_index.append(edge_list_full.index(e[::-1]))
			else:
				raise ValueError('edge not in edge_list')
		edge_color = nx.to_pandas_edgelist(G)['weight'][edge_list_index]
	else:
		edge_color = nx.to_pandas_edgelist(G)['weight']
	if 'ax' not in kwargs:
		kwargs['ax'] = plt.gca()
		kwargs_extended['ax'] = kwargs['ax']
	else:
		kwargs_extended['ax'] = kwargs['ax']

	nx.draw(
		G, pos=graph.node_positions, edge_color=edge_color,
		**kwargs
	)

	if kwargs_extended['show_colormap']:
		# draw colormap
		ax = plt.gca()
		sm = cm.ScalarMappable(
			cmap=kwargs['edge_cmap'], norm=plt.Normalize(vmin=kwargs['edge_vmin'], vmax=kwargs['edge_vmax'])
		)
		sm._A = []
		cbar = plt.colorbar(
			sm, pad=kwargs_extended['cmap_pad'], label=kwargs_extended['color_label'], extend=kwargs_extended['extend'],
			ax=kwargs_extended['ax']
		)