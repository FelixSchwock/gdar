import numpy as np
import copy
from scipy import spatial

class Graph:
	"""
	Generic graph class. Different ways of representing a graph
	"""
	def __init__(self):
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
		self.lmda_L1_low = None
		self.V_L1_low = None
		self.node_positions = None
		self.node_names = None

	def generate_from_edge_list(self, edge_list):
		"""
		edge_list should have the structure [(from_node, to_node, {'weight': w})].
		Node labels should start at 0.
		The reference orientation for B1 is from the node with lower index to the node with higher index
		"""
		# get number of nodes
		nodes = []
		for n in edge_list:
			nodes.append(n[0])
			nodes.append(n[1])
		self.N = np.max(nodes) + 1
		self.nodes = np.unique(nodes)

		# edges:
		self.E = len(edge_list)

		# node to incidence matrix B1, edge signal s1, and edge list
		self.B1 = np.zeros((self.N, self.E))
		self.s1 = np.zeros(self.E)
		self.edge_list = []
		for i, edge in enumerate(edge_list):
			if edge[0] < edge[1]:
				self.B1[edge[0],i] = -1
				self.B1[edge[1],i] = 1
				self.s1[i] = edge[2]['weight']
				self.edge_list.append(edge)
			else:
				self.B1[edge[1],i] = -1
				self.B1[edge[0],i] = 1
				self.s1[i] = -edge[2]['weight']
				self.edge_list.append((edge[1], edge[0], {'weight': -edge[2]['weight']}))

		# combinatorial Laplacian
		self.L = self.B1 @ self.B1.T

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
		Add edges to existing graph
		"""
		edge_list_new = copy.deepcopy(self.edge_list)
		for edge in edge_list:
			edge_list_new.append((edge[0], edge[1], {'weight': 1}))

		self.generate_from_edge_list(edge_list_new)

	def edge_list_correct_direction(self, edge_list=None):
		"""
		flips direction of arrow is edge_weight is negative. Useful for plotting edge signals
		"""
		if edge_list is None:
			edge_list = self.edge_list

		new_edge_list = copy.deepcopy(edge_list)
		for i in range(len(edge_list)):
			if edge_list[i][2]['weight'] < 0:
				from_edge = edge_list[i][1]
				to_edge = edge_list[i][0]
				new_edge_list[i] = (from_edge, to_edge, {'weight': -new_edge_list[i][2]['weight']})

		return new_edge_list

	def get_diameter(self):
		"""
		Compute diameter of graph
		:return: diameter
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
		self.L = self.B1 @ self.B1.T
		return self.L

	def get_fourier_basis(self, L=None):
		"""
		Compute Fourier basis for the graph
		:return: (eigenvalues, eigenvectors)
		"""
		if self.L is None:
			L = self.get_graph_laplacian()
		w, V = np.linalg.eigh(L)
		return w, V

	def hodge_decomposition(self):
		if self.B1 is not None:
			w_grad, V_grad = np.linalg.eigh(self.B1 @ self.B1.T)
			self.w_grad = w_grad[1:]
			self.V_grad = self.B1.T @ V_grad[:, 1:]

		if self.B2 is not None:
			# this lifts down from triangle domain to edge domain but doen't create an orthorgonal basis
			# self.w_curl, V_curl = np.linalg.eigh(self.graph.B2.T @ self.graph.B2)
			# self.V_curl = self.graph.B2 @ V_curl

			# this is in the the edge domain and creates an orthogonal basis
			self.w_curl, self.V_curl = np.linalg.eigh(self.B2 @ self.B2.T)
			# only keep non-zero eigenvalues
			non_zero_idx = self.w_curl > 1e-8
			self.w_curl = self.w_curl[non_zero_idx]
			self.V_curl = self.V_curl[:, non_zero_idx]

		return self.w_grad, self.V_grad, self.w_curl, self.V_curl


	def get_lower_hodge_laplacian(self):
		if self.B1 is not None:
			self.L1l = self.B1.T @ self.B1
			return self.L1l
		else:
			print('Generate graph first')
			return None

	def get_upper_hodge_laplacian(self):
		self.L1u = self.B2 @ self.B2.T
		return self.L1u

	def get_hodge_laplacian(self):
		self.L1 = self.B1.T @ self.B1 + self.B2 @ self.B2.T
		return self.L1

	def delauney_triangulation(self, node_positions):
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
		self.node_positions = node_positions
		self.node_names = node_names

	def get_subgraph_edges_from_nodes(self, node_list):
		"""
		Get edges that are connected to nodes in node_list
		param: node_list: list of nodes

		return: edge_list: list of edges
		"""
		edge_list = []
		for k, edge in enumerate(self.edge_list):
			if edge[0] in node_list or edge[1] in node_list:
				edge_list.append(k)
		return edge_list


