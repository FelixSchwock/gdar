from matplotlib import pyplot as plt
import numpy as np
import networkx as nx
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import ArrowStyle

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
	new_cmap = LinearSegmentedColormap.from_list(
		'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
		cmap(np.linspace(minval, maxval, n)))
	return new_cmap

def plot_graph(graph, directed=True, **kwargs):
	"""
	:param graph:
	:param directed:
	:param kwargs:
	:return:
	"""

	edge_list = graph.edge_list

	if not directed:
		G = nx.Graph(edge_list)
	else:
		edge_list_direction_corr = graph.edge_list_correct_direction(edge_list)
		G = nx.DiGraph(edge_list_direction_corr)

	kwargs_extended = {}

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

def plot_flow_graph(flow_signal, directed=True, **kwargs):
	"""
	:param flow_signal: FlowSignal object
		if flow contains more than one snapshot, plot fist snapshot
	:param kwargs:
	:return:
	"""
	flow_signal.set_edge_weights()
	graph = flow_signal.graph
	edge_list = graph.edge_list

	if not directed:
		G = nx.Graph(edge_list)
	else:
		edge_list_direction_corr = graph.edge_list_correct_direction(edge_list)
		G = nx.DiGraph(edge_list_direction_corr)

	kwargs_extended = {}

	if 'edge_cmap' not in kwargs:
		kwargs['edge_cmap'] = cm.bwr
	if 'vlim_quantile' in kwargs:
		vlim = np.quantile(np.abs(flow_signal.f[:,0]), kwargs['vlim_quantile'])
		kwargs['edge_vmin'] = -vlim
		kwargs['edge_vmax'] = vlim
		del kwargs['vlim_quantile']
	if 'edge_vmin' not in kwargs:
		kwargs['edge_vmin'] = np.min(flow_signal.f[:,0])
	if 'edge_vmax' not in kwargs:
		kwargs['edge_vmax'] = np.max(flow_signal.f[:,0])
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

class NetworkAnimation:
	def __init__(self, S1_mtx, graph, electrode_pos, electrode_idx, stim1_idx, stim2_idx, m1_nodes, s1_nodes,
	             vmin=None, vmax=None, cmap='default', node_color='lightgray', stim_node_color='black', figsize=(14,8),
	             annotation=True):
		self.graph = graph
		self.S1_mtx = S1_mtx
		self.electrode_pos = electrode_pos
		self.electrode_idx = electrode_idx
		self.stim1_idx = stim1_idx
		self.stim2_idx = stim2_idx
		self.s1_nodes = s1_nodes
		self.m1_nodes = m1_nodes
		self.annotation = annotation

		if cmap == 'default':
			cmap = cm.seismic
			colors = cmap(np.linspace(0.5, 1, cmap.N // 2))
			self.cmap = LinearSegmentedColormap.from_list('Upper Half', colors)
		else:
			self.cmap = cmap

		self.graph.set_edge_signal(self.S1_mtx[:,0])
		self.node_color = node_color
		self.stim_node_color = stim_node_color
		direction_corr_edge_list = self.graph.edge_list_correct_direction()
		G1 = nx.DiGraph(direction_corr_edge_list)
		#G1 = nx.Graph(direction_corr_edge_list)

		if vmin is None:
			self.vmin = 0
		else:
			self.vmin = vmin
		if vmax is None:
			self.vmax = np.quantile(abs(self.S1_mtx), 0.95)
		else:
			self.vmax = vmax

		edge_width = np.ones(len(direction_corr_edge_list)) * 3.0
		for k, e in enumerate(G1.edges):
			if (e[0] in self.m1_nodes and e[1] in self.s1_nodes) or \
				(e[0] in self.s1_nodes and e[1] in self.m1_nodes):
				edge_width[k] = 8.0

		self.arrow_style = ArrowStyle.CurveFilledB(head_length=0.8, head_width=0.3)

		if figsize is not None:
			self.fig, self.ax = plt.subplots(figsize=figsize)
		nx.draw(G1, pos=self.electrode_pos, edge_color = nx.to_pandas_edgelist(G1)['weight'],
		        width=edge_width, edge_cmap=self.cmap, node_size=300, node_color=self.node_color,
		        edge_vmin=self.vmin, edge_vmax=self.vmax, arrowstyle=self.arrow_style)

		self.stim_nodes = []
		if self.stim1_idx is not None:
			self.stim_nodes.append(self.stim1_idx)
		if self.stim2_idx is not None:
			self.stim_nodes.append(self.stim2_idx)

		if len(self.stim_nodes) > 0:
			nodes = nx.draw_networkx_nodes(G1, pos=self.electrode_pos, nodelist=self.stim_nodes,
			                               node_size=700, node_color=self.stim_node_color, node_shape='X', linewidths=3)

		if self.electrode_idx is not None:
			if len(self.electrode_idx) == len(self.electrode_pos):
				labels = nx.draw_networkx_labels(G1, pos=self.electrode_pos, labels=self.electrode_idx)

		if self.annotation:
			sm = plt.cm.ScalarMappable(cmap=self.cmap, norm=plt.Normalize(vmin=self.vmin, vmax=self.vmax))
			sm._A = []
			cbar = plt.colorbar(sm, pad=-0.05, label='normalized flow magnitude', extend='max')
			plt.title(f'Stim + 0 ms')

	def anim_init(self):
		pass

	def animate(self, frame):
		self.ax.clear()
		self.graph.set_edge_signal(self.S1_mtx[:,frame])
		direction_corr_edge_list = self.graph.edge_list_correct_direction()
		G1 = nx.DiGraph(direction_corr_edge_list)

		edge_width = np.ones(len(direction_corr_edge_list)) * 3.0
		for k, e in enumerate(G1.edges):
			if (e[0] in self.m1_nodes and e[1] in self.s1_nodes) or \
				(e[0] in self.s1_nodes and e[1] in self.m1_nodes):
				edge_width[k] = 8.0

		nx.draw(G1, pos=self.electrode_pos, edge_color = nx.to_pandas_edgelist(G1)['weight'],
		        width=edge_width, edge_cmap=self.cmap, node_size=300, node_color=self.node_color,
		        edge_vmin=self.vmin, edge_vmax=self.vmax, arrowstyle=self.arrow_style)

		if len(self.stim_nodes) > 0:
			nodes = nx.draw_networkx_nodes(G1, pos=self.electrode_pos, nodelist=self.stim_nodes,
			                               node_size=700, node_color=self.stim_node_color, node_shape='X', linewidths=3)

		if self.electrode_idx is not None:
			if len(self.electrode_idx) == len(self.electrode_pos):
				labels = nx.draw_networkx_labels(G1, pos=self.electrode_pos, labels=self.electrode_idx)

		if self.annotation:
			plt.title(f'Stim + {frame} ms')