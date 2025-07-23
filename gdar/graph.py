import numpy as np
import copy
from scipy import spatial

class Graph:
	"""
	Generic graph class. Different ways of representing a graph
	"""
	def __init__(self):
		"""
		The graph objecthas the following attributes:
		- B1: node to edge incident matrix
		- B2: edge to triangle incident matrix
		- N: number of nodes
		- E: number of edges
		- A: adjacency matrix
		- D: degree matrix
		- edge_list: list of edges
		- nodes: list of nodes
		- neighbors: list of neighbors for each node
		- L: Combinatorial Laplacian
		- L1l: lower Hodge-1 Laplacian
		- L1u: upper Hodge-1 Laplacian
		- L1: Hodge-1 Laplacian
		- node_positions: dictionary with node positions
		- node_names: list of node names
		"""

		self.B1 = None  # node to edge incident matrix
		self.B2 = None  # edge to triangle incident matrix
		self.N = None  # number of nodes
		self.E = None  # number of edges
		self.A = None  # adjacency matrix
		self.D = None  # degree matrix
		self.edge_list = None  # list of edges
		self.nodes = None  # list of nodes
		self.neighbors = None  # lift of neighbors for each node
		self.L = None  # Combinatorial Laplacian
		self.L1l = None  # lower Hodge-1 Laplacian
		self.L1u = None  # upper Hodge-1 Laplacian
		self.L1 = None  # Hodge-1 Laplacian
		self.node_positions = None
		self.node_names = None

	def generate_from_edge_list(self, edge_list):
		"""
		Generate graph from edge list. Each entry of the edge_list should be a tuple of the form (from_node, to_node).
		Node labeling should start at 0 and be consecutive integers. Which node is the from_node and which is the
		to_node is arbitrary for the GDAR model. The functions will set the number of nodes `N` and edges `E`, the node
		list `nodes`, node to edge incidence matrix `B1`, the adjacency matrix `A`,	the degree matrix `D`, the
		combinatorial Laplacian `L`, the edge list `edge_list`, the list of unique `nodes`, and the `neighbors`.
		The edge list will be sorted such that the smaller node index is always the first element in the tuple.
		dictionary.

		Parameters:
			edge_list (list): List of edges, where each edge is a tuple (from_node, to_node).
		"""
		# get number of nodes and list of unique nodes
		nodes = []
		for n in edge_list:
			nodes.append(n[0])
			nodes.append(n[1])
		self.N = np.max(nodes) + 1
		self.nodes = np.unique(nodes)

		# edges:
		self.E = len(edge_list)

		# node to incidence matrix B1, and edge list
		self.B1 = np.zeros((self.N, self.E))
		self.edge_list = []
		for i, edge in enumerate(edge_list):
			if edge[0] < edge[1]:
				self.B1[edge[0],i] = -1
				self.B1[edge[1],i] = 1
				self.edge_list.append(edge)
			else:
				self.B1[edge[1],i] = -1
				self.B1[edge[0],i] = 1
				self.edge_list.append((edge[1], edge[0]))

		# combinatorial Laplacian
		self.get_graph_laplacian()

		# degree matrix
		self.D = np.diag(np.diag(self.L))

		# Adjacency matrix
		self.A = self.D - self.L

		# neighbor dictionary
		neighbors = {}
		for node in self.nodes:
			n_list = []
			for edge in self.edge_list:
				if node in edge:
					if node == edge[0]:
						n_list.append(edge[1])
					else:
						n_list.append(edge[0])
			neighbors[node] = n_list
		self.neighbors = neighbors

	def add_edges(self, edge_list):
		"""
		Add edges to existing graph.

		Parameters:
			edge_list (list): List of edges to add, where each edge is a tuple (from_node, to_node).
			Note that the edge list should not contain duplicate edges.
		"""
		edge_list_new = copy.deepcopy(self.edge_list)
		for edge in edge_list:
			edge_list_new.append((edge[0], edge[1]))

		self.generate_from_edge_list(edge_list_new)

	def nn_graph(self, node_positions, n_neighbors=8):
		"""
		Creates a nearest neighbor graph if either x and y or x, y, and z coordinates are given.

		Parameters:
			node_positions (dict): Dictionary with node positions. Each entry should be of the form `node_id: pos`,
				where `pos` is a numpy array of shape (2,) for 2D or (3,) for 3D.
			n_neighbors (int): Number of nearest neighbors to connect each node to. Default is 8.

		Returns:
			self: The graph object with the generated nearest neighbor graph.
		"""
		edge_list = []
		self.node_positions = node_positions

		for v1 in node_positions:
			pos_v1 = node_positions[v1]

			# for each node compute distance to all other nodes and sort them
			dist_to_v1 = []
			v2_list = []
			for v2 in node_positions:
				pos_v2 = node_positions[v2]
				dist = np.sum((pos_v1 - pos_v2)**2)
				dist_to_v1.append(dist)
				v2_list.append(v2)
			idx_sort = np.argsort(dist_to_v1)
			v2_list = np.array(v2_list)

			# extract n_neighbors closest neighbors
			v1_neighbors = v2_list[idx_sort[1:n_neighbors+1]]

			# add edges to edge list
			for v2 in v1_neighbors:
				if (np.min([v1, v2]), np.max([v1, v2])) not in edge_list:
					edge_list.append((np.min([v1, v2]), np.max([v1, v2])))

		self.generate_from_edge_list(edge_list)
		return self

	def proximity_graph(self, node_positions, dist_th=5):
		"""
		Creates graph based on node proximity. All nodes that are less or equal than `dist_th` away from each other are
		connected.

		Parameters:
			node_positions (dict): Dictionary with node positions. Each entry should be of the form `node_id: pos`,
				where `pos` is a numpy array of shape (2,) for 2D or (3,) for 3D.
			dist_th (float): Distance threshold. Nodes that are closer than this distance will be connected.

		Returns:
			self: The graph object with the generated proximity graph.
		"""
		edge_list = []
		self.node_positions = node_positions

		for v1 in node_positions:
			pos_v1 = node_positions[v1]

			v2_list = []
			for v2 in node_positions:
				pos_v2 = node_positions[v2]
				dist = np.sqrt(np.sum((pos_v1 - pos_v2)**2))
				if dist <= dist_th and dist != 0:
					v2_list.append(v2)

			for v2 in v2_list:
				if (np.min([v1, v2]), np.max([v1, v2])) not in edge_list:
					edge_list.append((np.min([v1, v2]), np.max([v1, v2])))

		self.generate_from_edge_list(edge_list)
		return self

	def custom_proximity_graph(self, node_positions, dist=None):
		"""
		Creates graph based on node proximity, where the distance threshold can be specified for each node individually.

		Parameters:
			node_positions (dict): Dictionary with node positions. Each entry should be of the form `node_id: pos`,
				where `pos` is a numpy array of shape (2,) for 2D or (3,) for 3D.
			dist (tuple or dict): Distance threshold. If a tuple is provided, it will be used to assign nodes to two
				different distance thresholds. If a dictionary is provided, it should contain the distance thresholds
				for each node. The dictionary should be of the form:
				```
				dist_dct = {
					dist1: [d1_n1, d1_n2, ...],
					dist2: [d2_n1, d2_n2, ...],
					...
				}
				```
				where `dist1`, `dist2`, etc. are the distance thresholds and `d1_n1`, `d1_n2`, etc. are the nodes that

		Returns:
			self: The graph object with the generated proximity graph.
		"""
		self.node_positions = node_positions
		init_dist_dct = False
		if type(dist) is tuple:
			dist_dct = {
				dist[0]: [],
				dist[1]: []
			}
			init_dist_dct = True
		else:
			dist_dct = dist

		# assign nodes automatically to distance thresholds based on their distance to nearest neighbor
		if init_dist_dct:
			for v1 in node_positions:
				v1_neighbor_dist = []
				for v2 in node_positions:
					d = np.sqrt(np.sum((node_positions[v1] - node_positions[v2])**2))
					if d > 0:
						v1_neighbor_dist.append(d)
				if np.min(v1_neighbor_dist) < dist[0]:
					dist_dct[dist[0]].append(v1)
				else:
					dist_dct[dist[1]].append(v1)

		edge_list = []
		for dist_th in dist_dct:
			for v1 in dist_dct[dist_th]:
				pos_v1 = node_positions[v1]

				v2_list = []
				for v2 in node_positions:
					pos_v2 = node_positions[v2]
					dist = np.sqrt(np.sum((pos_v1 - pos_v2)**2))
					if dist <= dist_th and dist != 0:
						v2_list.append(v2)

				for v2 in v2_list:
					if (np.min([v1, v2]), np.max([v1, v2])) not in edge_list:
						edge_list.append((np.min([v1, v2]), np.max([v1, v2])))

		self.generate_from_edge_list(edge_list)
		return self

	def get_diameter(self):
		"""
		Compute diameter of graph

		Returns:
			int: Diameter of the graph, which is the maximum distance between any two nodes in the graph.
		"""

		# compute exponential of adjacency matrix
		A = copy.deepcopy(self.A)
		A_sum = copy.deepcopy(self.A)
		k = 1
		while np.any(A_sum == 0):
			A = A @ self.A
			A_sum += A
			k += 1
		return k

	def get_graph_laplacian(self):
		"""
		Compute the combinatorial Laplacian of the graph, which is defined as `L = B1 @ B1.T`, where `B1` is the node
			to edge incidence matrix.

		Returns:
			np.ndarray: Combinatorial Laplacian matrix `L`
		"""

		self.L = self.B1 @ self.B1.T
		return self.L

	def get_lower_hodge_laplacian(self):
		"""
		Compute the lower Hodge-1 Laplacian of the graph, which is defined as `L1l = B1.T @ B1`, where `B1` is the
			node to edge incidence matrix.

		Returns:
			np.ndarray: Lower Hodge-1 Laplacian matrix `L1l`
		"""

		if self.B1 is not None:
			self.L1l = self.B1.T @ self.B1
			return self.L1l
		else:
			print('Generate graph first')
			return None

	def get_upper_hodge_laplacian(self):
		"""
		Compute the upper Hodge-1 Laplacian of the graph. Note that the edge-to-triangle incidence matrix `B2` has to be
		set before calling this function. This can be done by calling `delaunay_triangulation` or `B2_all_triangles`.

		Returns:
			np.ndarray: Upper Hodge-1 Laplacian matrix `L1u`
		"""

		self.L1u = self.B2 @ self.B2.T
		return self.L1u

	def get_hodge_laplacian(self):
		"""
		Compute the full Hodge-1 Laplacian of the graph. Note that the edge-to-triangle incidence matrix `B2` has to be
		set before calling this function. This can be done by calling `delaunay_triangulation` or `B2_all_triangles`.

		Returns:
			np.ndarray: Hodge-1 Laplacian matrix `L1`
		"""

		self.L1 = self.B1.T @ self.B1 + self.B2 @ self.B2.T
		return self.L1

	def delaunay_triangulation(self, node_positions):
		"""
		Compute the Delaunay triangulation of the graph based on the node positions.

		Parameters:
			node_positions (dict): Dictionary with node positions. Each entry should be of the form `node_id: pos`,
				where `pos` is a numpy array of shape (2,) for 2D or (3,) for 3D.
		Returns:
			np.ndarray: Edge to triangle incidence matrix `B2`
		"""

		node_pos_arr = []
		self.node_positions = node_positions
		for n in self.node_positions:
			node_pos_arr.append(self.node_positions[n])
		node_pos_arr = np.array(node_pos_arr)
		tri = spatial.Delaunay(node_pos_arr)

		# obtain edge to triangle incident matrix B2 from simplices
		# if triangle includes edge that is not included in B1, remove triangle
		B2 = []
		edge_array = []
		for e in self.edge_list:
			edge_array.append([e[0], e[1]])

		for triangle in tri.simplices:
			# obtain all ordered pairs for each traingle
			edges = [
				[triangle[0], triangle[1]],
				[triangle[1], triangle[0]],
				[triangle[1], triangle[2]],
				[triangle[2], triangle[1]],
				[triangle[2], triangle[0]], # important to start with 2 here so that we follow triangular flow
				[triangle[0], triangle[2]],
			]
			B2_col = np.zeros(self.E)
			cnt = 0
			for i, e in enumerate(edges):
				if e in edge_array:
					edge_idx = edge_array.index(e)
					B2_col[edge_idx] = (-1)**i
					cnt += 1
			if cnt != 3:
				continue
			else:
				B2.append(B2_col)
		self.B2 = np.array(B2).T
		return self.B2

	def B2_all_triangles(self):
		"""
		Compute the edge to triangle incidence matrix B2 using all possible triangles in the graph.
		This function checks all edges and finds all triangles that can be formed by the edges.

		Returns:
			np.ndarray: Edge to triangle incidence matrix `B2`
		"""
		triangles = []
		for e in self.edge_list:
			i = e[0]
			j = e[1]
			for k in self.neighbors[i]:
				if k in self.neighbors[j]:
					if [i, j, k] not in triangles and [j, i, k] not in triangles and [i, k, j] not in triangles and [j, k, i] not in triangles and [k, i, j] not in triangles and [k, j, i] not in triangles:
						triangles.append([i, j, k])

		B2 = []
		edge_array = []
		for e in self.edge_list:
			edge_array.append([e[0], e[1]])

		for triangle in triangles:
			# obtain all ordered pairs for each traingle
			edges = [
				[triangle[0], triangle[1]],
				[triangle[1], triangle[0]],
				[triangle[1], triangle[2]],
				[triangle[2], triangle[1]],
				[triangle[2], triangle[0]], # important to start with 2 here so that we follow triangular flow
				[triangle[0], triangle[2]],
			]
			B2_col = np.zeros(self.E)
			cnt = 0
			for i, e in enumerate(edges):
				if e in edge_array:
					edge_idx = edge_array.index(e)
					B2_col[edge_idx] = (-1)**i
					cnt += 1
			if cnt != 3:
				continue
			else:
				B2.append(B2_col)
		self.B2 = np.array(B2).T
		return self.B2

	def set_node_position_name(self, node_positions, node_names=None):
		"""
		Set node positions and optionally node names for the graph.

		Parameters:
			node_positions (dict): Dictionary with node positions. Each entry should be of the form `node_id: pos`,
				where `pos` is a numpy array of shape (2,) for 2D or (3,) for 3D.
			node_names (list, optional): List of node names. If provided, should have the same length as the number of
				nodes in the graph. Default is None.
		"""
		self.node_positions = node_positions
		self.node_names = node_names

	def get_subgraph_edges_from_nodes(self, node_list):
		"""
		Returns subgraph in the form of a list of edges. Only edges that are connected to nodes in node_list are included.

		Parameters:
			node_list (list): List of node indices for which the subgraph edges should be returned.
		Returns:
			list: List of edge indices that are connected to the nodes in node_list.
		"""
		edge_list = []
		for k, edge in enumerate(self.edge_list):
			if edge[0] in node_list or edge[1] in node_list:
				edge_list.append(k)
		return edge_list


