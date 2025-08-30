from matplotlib import pyplot as plt
import numpy as np
import networkx as nx
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap
from scipy.interpolate import griddata
from matplotlib.patches import FancyArrowPatch


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

def plot_graph(graph, directed=True, projection=None, **kwargs):
	"""
	Plot graph using NetworkX.

	Parameters:
		graph (:class:`gdar.graph.Graph`): Graph object containing edge_list and node_positions.
		directed (bool): If True, plot as directed graph; otherwise, as undirected.
		projection (str): Projection if x, y, z coordinates are given in node_positions. Options are 'xy', 'xz', 'yz',
			or None (default). If None, the first two dimensions (xy) are used.
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

	# determine positions for nodes based on projection
	if projection is None or projection == 'xy' or len(graph.node_positions[0]) == 2:
		pos = {node: (graph.node_positions[node][0], graph.node_positions[node][1]) for node in graph.node_positions}
	elif projection == 'yz':
		pos = {node: (graph.node_positions[node][1], graph.node_positions[node][2]) for node in graph.node_positions}
	elif projection == 'xz':
		pos = {node: (graph.node_positions[node][0], graph.node_positions[node][2]) for node in graph.node_positions}
	else:
		raise ValueError("Invalid projection. Choose from 'xy', 'xz', 'yz', or None.")

	nx.draw(G, pos=pos, **kwargs)

def plot_flow_graph(graph, flow_vector, directed=True, projection=None, **kwargs):
	"""
	Plot flow graph using NetworkX.

	Parameters:
		graph (:class:`gdar.graph.Graph`): Graph object containing edge_list and node_positions.
		flow_vector (np.ndarray): Flow vector with flow values for each edge.
		directed (bool): If True, plot as directed graph; otherwise, as undirected.
		projection (str): Projection if x, y, z coordinates are given in node_positions. Options are 'xy', 'xz', 'yz',
			or None (default). If None, the first two dimensions (xy) are used.
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

	# determine positions for nodes based on projection
	if projection is None or projection == 'xy' or len(graph.node_positions[0]) == 2:
		pos = {node: (graph.node_positions[node][0], graph.node_positions[node][1]) for node in
			   graph.node_positions}
	elif projection == 'yz':
		pos = {node: (graph.node_positions[node][1], graph.node_positions[node][2]) for node in
			   graph.node_positions}
	elif projection == 'xz':
		pos = {node: (graph.node_positions[node][0], graph.node_positions[node][2]) for node in
			   graph.node_positions}
	else:
		raise ValueError("Invalid projection. Choose from 'xy', 'xz', 'yz', or None.")

	nx.draw(
		G, pos=pos, edge_color=edge_color,
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

def plot_flow_field(graph, flow_vector, projection='xy', plot_potentials=True, origin='lower', ax=None,
					cmap='bwr_r', vmin=-0.5, vmax=0.5, interpolation='bilinear', node_size=30, node_color='lightgray',
					flow_filed_scaling_factor=15, arrowstyle='-|>', mutation_scale=8, arrow_color='black', linewidth=2,):
	"""
	Plot flow and potential field from flow vector. The potential field is computed as graph.B1 @ flow_vector and plotted
	as a background using imshow. The flow field is computed as the average flow at each node.

	Parameters:
		graph (:class:`gdar.graph.Graph`): Graph object containing edge_list and node_positions.
		flow_vector (np.ndarray): Flow vector with flow values for each edge.
		projection (str): Projection if x, y, z coordinates are given in node_positions. Options are 'xy', 'xz', 'yz',
			or None (default). If None, the first two dimensions (xy) are used.
		plot_potentials (bool): If True, plot potential field as background (default: True).
		origin (str): Origin parameter for imshow (default: 'lower').
		ax (matplotlib axis): Matplotlib axis to draw on (default: None, creates new figure).
		cmap (str or Colormap): Colormap for potential field (default: 'bwr_r').
		vmin (float): Minimum value for potential field colormap (default: -0.5).
		vmax (float): Maximum value for potential field colormap (default: 0.5).
		interpolation (str): Interpolation method for imshow (default: 'bilinear').
		node_size (int): Size of nodes in the graph (default: 30).
		node_color (str or list): Color of nodes in the graph (default: 'lightgray').
		flow_filed_scaling_factor (float): Scaling factor for flow field arrows (default: 15).
		arrowstyle (str): Arrow style for flow field arrows (default: '-|>').
		mutation_scale (int): Mutation scale for flow field arrows (default: 8).
		arrow_color (str): Color of flow field arrows (default: 'black').
		linewidth (float): Line width of flow field arrows (default: 2).
	Returns:
		tuple or array: If plot_potentials is True, returns a tuple (potentials, flow_vector) where potentials is the
			potential field and flow_vector is the flow vector at each node. If plot_potentials is False, returns
			only the flow_vector at each node.
	"""

	# flow field
	flow_field_dct = {}

	if ax is None:
		fig, ax = plt.subplots(figsize=(8, 8))

	# determine positions for nodes based on projection
	if projection is None or projection == 'xy' or len(graph.node_positions[0]) == 2:
		pos = {node: np.array([graph.node_positions[node][0], graph.node_positions[node][1]]) for node in graph.node_positions}
	elif projection == 'yz':
		pos = {node: np.array([graph.node_positions[node][1], graph.node_positions[node][2]]) for node in graph.node_positions}
	elif projection == 'xz':
		pos = {node: np.array([graph.node_positions[node][0], graph.node_positions[node][2]]) for node in graph.node_positions}
	else:
		raise ValueError("Invalid projection. Choose from 'xy', 'xz', 'yz', or None.")

	if plot_potentials:
		# potential field
		potentials = graph.B1 @ flow_vector

		# get node_positions as array
		x_coords = np.array([pos[n][0] for n in pos])
		y_coords = np.array([pos[n][1] for n in pos])

		# interpolate positions using grdiddata
		xi = np.linspace(np.min(x_coords), np.max(x_coords), 100)
		yi = np.linspace(np.min(y_coords), np.max(y_coords), 100)
		X, Y = np.meshgrid(xi, yi)
		Z = griddata((x_coords, y_coords), potentials, (X, Y), method='cubic')

		# create figure
		plt.imshow(Z, extent=(np.min(x_coords), np.max(x_coords), np.min(y_coords), np.max(y_coords)), origin=origin,
				   cmap=cmap, vmin=vmin, vmax=vmax, interpolation=interpolation)

	# plot node graph on top
	plot_graph(graph, ax=ax, node_color=node_color, node_size=node_size, width=0, directed=False)

	for n in range(graph.N):
		neighbors = graph.neighbors[n]
		flow_vectors = 0 + 0j
		for neighbor in neighbors:
			# identify edge angle
			if (n, neighbor) in graph.edge_list or (n, neighbor, {'weight': 1}) in graph.edge_list:
				tail_node = n
				head_node = neighbor
			else:
				tail_node = neighbor
				head_node = n
			direction_cart = pos[head_node] - pos[tail_node]
			direction_angle = np.angle(direction_cart[0] + 1j * direction_cart[1])
			edge_idx = graph.edge_list.index((tail_node, head_node))
			flow_vectors += flow_vector[edge_idx] * np.exp(1j * direction_angle) / len(neighbors)
		flow_field_dct[n] = {
			'flow': flow_vectors,
			'start': pos[n]
		}

	# plot flow vector field
	for n in range(graph.N):
		flow_vector = flow_field_dct[n]['flow']
		start = flow_field_dct[n]['start']
		ax.add_patch(
			FancyArrowPatch(
				start,
				start + (np.real(flow_vector) * flow_filed_scaling_factor, np.imag(flow_vector) * flow_filed_scaling_factor),
				arrowstyle=arrowstyle,
				mutation_scale=mutation_scale,
				color=arrow_color,
				linewidth=linewidth,
				zorder=10
			)
		)

	if plot_potentials:
		plt.colorbar(label='potential', pad=-0.05)
		return potentials, flow_vectors
	else:
		return flow_vectors