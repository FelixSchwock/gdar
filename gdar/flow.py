import numpy as np
from scipy import linalg
from copy import deepcopy

class FlowSignal():
	def __init__(self, graph, f=None):
		"""
		:param f: numpy.ndarray
			E x T array, where E are the number of edges and T the number of time steps
		"""
		self.graph = deepcopy(graph)
		self.f = f
		self.w_grad = None
		self.V_grad = None
		self.w_curl = None
		self.V_curl = None
		self.w_harm = None
		self.V_harm = None

	def set_flow_signal(self, f):
		self.f = f

	def set_edge_weights(self, edge_weights=None):
		if edge_weights is None:
			edge_weights = self.f[:,0]
		elif len(edge_weights) != len(self.graph.edge_list):
			raise TypeError('edge signal s1 must have same length as edge_list')

		for i in range(len(edge_weights)):
			self.graph.edge_list[i][2]['weight'] = edge_weights[i]

	def set_graph(self, graph):
		self.graph = graph

	def hodge_decomposition(self, mode='full'):
		if self.graph.B1 is not None and mode == 'grad':
			w_grad, V_grad = np.linalg.eigh(self.graph.B1 @ self.graph.B1.T)
			self.w_grad = w_grad[1:]
			self.V_grad = self.graph.B1.T @ V_grad[:, 1:]

			return self.w_grad, self.V_grad,

		if self.graph.B2 is not None and mode == 'curl':
			# this lifts down from triangle domain to edge domain but doen't create an orthorgonal basis
			# self.w_curl, V_curl = np.linalg.eigh(self.graph.B2.T @ self.graph.B2)
			# self.V_curl = self.graph.B2 @ V_curl

			# this is in the the edge domain and creates an orthogonal basis
			self.w_curl, self.V_curl = np.linalg.eigh(self.graph.B2 @ self.graph.B2.T)
			# only keep non-zero eigenvalues
			non_zero_idx = self.w_curl > 1e-8
			self.w_curl = self.w_curl[non_zero_idx]
			self.V_curl = self.V_curl[:, non_zero_idx]

			return self.w_curl, self.V_curl

		if self.graph.B1 is not None and self.graph.B2 is not None and mode == 'full':
			L1 = self.graph.get_hodge_laplacian()
			w, V = np.linalg.eigh(L1)
			zero_indices = np.where(np.abs(w) < 1e-8)[0]
			self.V_harm = V[:, zero_indices]
			self.w_harm = np.zeros(len(zero_indices))

			grad_indices = np.where(np.sum(np.abs(self.graph.B1 @ V), axis=0) > 1e-8)[0]
			self.w_grad = w[grad_indices]
			self.V_grad = V[:, grad_indices]

			# curl indices are all remaining indices
			curl_indices = np.setdiff1d(np.arange(len(w)), np.concatenate((zero_indices, grad_indices)))
			self.w_curl = w[curl_indices]
			self.V_curl = V[:, curl_indices]

			return self.w_grad, self.V_grad, self.w_curl, self.V_curl, self.w_harm, self.V_harm

	def ft(self, f=None, idx=None, spectral_analysis=False):
		"""
		:param f: numpy.ndarray
			 E x T array, where E are the number of edges and T the number of time steps. If None
			 use self.f
		:param idx: int or array-like
			time indices for which FT is computed
		:return: flow spectrum (F_grad, F_curl)
		"""
		if f is None:
			f = self.f
		if idx is None:
			idx = np.arange(f.shape[1])

		if self.w_grad is None or self.w_curl is None:
			self.hodge_decomposition(mode='full')

		if self.V_grad is not None:
			if spectral_analysis:
				# normalize V_grad so that all spatial eigenvectors have unit norm
				V_grad_norm = np.linalg.norm(self.V_grad, axis=0)
				self.V_grad = self.V_grad / V_grad_norm[np.newaxis, :]
			F_grad = self.V_grad.T @ f[:,idx]
		else:
			F_grad = None

		if self.V_curl is not None:
			if spectral_analysis:
				# normalize V_curl so that all spatial eigenvectors have unit norm
				V_curl_norm = np.linalg.norm(self.V_curl, axis=0)
				self.V_curl = self.V_curl / V_curl_norm[np.newaxis, :]
			F_curl = self.V_curl.T @ f[:,idx]
		else:
			F_curl = None

		if self.V_harm is not None:
			if spectral_analysis:
				# normalize V_harm so that all spatial eigenvectors have unit norm
				V_harm_norm = np.linalg.norm(self.V_harm, axis=0)
				self.V_harm = self.V_harm / V_harm_norm[np.newaxis, :]
			F_harm = self.V_harm.T @ f[:,idx]
			self.w_harm = np.zeros(len(F_harm))
		else:
			F_harm = None

		return self.w_grad, F_grad, self.w_curl, F_curl, self.w_harm, F_harm

	def spectrogram(self, f=None, avg_win=1, spectral_analysis=False):
		if f is None:
			f = self.f

		w_grad, F_grad, w_curl, F_curl = self.ft(f, spectral_analysis=spectral_analysis)
		if F_grad is not None:
			Spec_grad = np.zeros((len(w_grad), int(np.ceil(f.shape[1] / avg_win))))
			F_grad_square = F_grad**2
			for i in range(int(f.shape[1] / avg_win)):
				Spec_grad[:,i] = np.mean(F_grad_square[:,i*avg_win:(i+1)*avg_win], axis=1)
		if F_curl is not None:
			Spec_curl = np.zeros((len(w_curl), int(np.ceil(f.shape[1] / avg_win))))
			F_curl_square = F_curl**2
			for i in range(int(f.shape[1] / avg_win)):
				Spec_curl[:,i] = np.mean(F_curl_square[:,i*avg_win:(i+1)*avg_win], axis=1)

		return w_grad, Spec_grad, w_curl, Spec_curl

	def joint_ft(self, f=None):
		if f is None:
			f = self.f

		T = f.shape[1]
		if T == 1:
			print('Need at least 2 time steps to compute joint FT')

		# TODO: implement
		pass

	def fir_filter(self, w_c, lmda_c):
		# TODO: implement
		pass

	def get_flow_adjaency_matrix(self, index=0, f=None):
		'''
		:param index: index if f is a matrix
		:param f: flow signal or matrix
		:return: flow signal represented as adjacency matrix
		'''
		if f is None:
			f = self.f[:,index]
		else:
			f = f[:,index]
		flow_adj = np.zeros((self.graph.N, self.graph.N))
		for i, e in enumerate(self.graph.edge_list):
			if np.sign(f[i]) == 1:
				flow_adj[e[0], e[1]] = f[i]
			else:
				flow_adj[e[1], e[0]] = -f[i]
		return flow_adj

	def communicability(self, flow_adj):
		"""
		:param flow_adj: flow signal represented as adjacency matrix
		:return: communicability matrix
		"""
		return linalg.expm(flow_adj)

	def tst_dictionary_basis(self, edge_subset_dict, freq_subset_dct, constraint='trade_off', lmda=0.5):
		"""
		Basis for Topological Slepian Transform (TST) for flow signals (1-complexes) on graphs.
		:param edge_subset_dict: dictionary
			keys location determinators (can be string or int)  and values are the corresponding edge indices. The
			structure should be as follows:
			{'grad': {'m1': [1, 2, 3], 's1': [4, 5, 6]}, 'curl': {'m1': [7, 8, 9], 's1': [10, 11]}}
		:param freq_subset_dct: dictionary
			keys are frequency bands (int; equivalent to scale parameter for wavelets) and values are the corresponding
			frequency indices. The structure should be as follows:
			{'grad': {1: [0, 1, 2], 2: [3, 4, 5]}, 'curl': {1: [0, 1, 2, 3], 2: [4, 5, 6]}}
		:param constraint: string
			either 'trade_off', 'spatial' or 'spectral'. For spatial and spectral, respective localization constraints
			are perfectly enforced. For trade_off, the localization constraints and traded off by the lambda parameter.
		:param lmda: float or tuple
			regularization parameter determining the weight assigned to spatial vs. spectral localization. If float, the
			same value is used for both grad and curl. If tuple, the first element is used for grad and the second for curl.
		:return: fancy wavelet basis for specified edge and frequency subsets as dictionary
		"""
		if isinstance(lmda, float):
			lmda_grad = lmda_curl = lmda
		elif isinstance(lmda, tuple):
			lmda_grad = lmda[0]
			lmda_curl = lmda[1]
		else:
			raise ValueError('lmda must be float or tuple')

		# define eigenvector matrix for full Hodge decomposition
		V = np.hstack((self.V_grad, self.V_curl, self.V_harm))

		tst_dictionary_basis = {}
		for grad_curl in ['grad', 'curl']:
			tst_dictionary_basis[grad_curl] = {}
			for loc in edge_subset_dict[grad_curl].keys():
				tst_dictionary_basis[grad_curl][loc] = {}
				# spacial localization matrix
				edge_subset_binary = np.zeros(self.graph.E)
				edge_subset_binary[edge_subset_dict[grad_curl][loc]] = 1
				D = np.diag(edge_subset_binary)
				for freq in freq_subset_dct[grad_curl].keys():
					tst_dictionary_basis[grad_curl][loc][freq] = {}
					if grad_curl == 'grad':
						freq_subset_binary = np.zeros(len(self.w_grad))
						freq_subset_binary[freq_subset_dct[grad_curl][freq]] = 1
						freq_subset_binary = np.concatenate(
							(freq_subset_binary,
							 np.zeros(self.V_curl.shape[1]),
							 np.zeros(self.V_harm.shape[1]))
						)
						curr_lmbda = lmda_grad
					elif grad_curl == 'curl':
						freq_subset_binary = np.zeros(len(self.w_curl))
						freq_subset_binary[freq_subset_dct[grad_curl][freq]] = 1
						freq_subset_binary = np.concatenate(
							(np.zeros(self.V_grad.shape[1]),
							 freq_subset_binary,
							 np.zeros(self.V_harm.shape[1]))
						)
						curr_lmbda = lmda_curl
					else:
						raise ValueError('grad_curl must be grad or curl')

					S = V @ np.diag(freq_subset_binary) @ V.T
					if constraint == 'spectral':
						w_phi, phi = np.linalg.eigh(S @ D @ S)
					elif constraint == 'spatial':
						w_phi, phi = np.linalg.eigh(D @ S @ D)
					elif constraint == 'trade_off':
						w_phi, phi = np.linalg.eigh(curr_lmbda * D + (1 - curr_lmbda) * S)
					else:
						raise ValueError('constraint must be trade_off, spatial or spectral')

					# fancy wavelet basis
					tst_dictionary_basis[grad_curl][loc][freq]['basis'] = phi[:, -1]

					# spectral center of mass
					w_grad, grad, w_curl, curl, w_harm, harm = self.ft(f=phi[:, -1].reshape(-1, 1), spectral_analysis=True)
					if grad_curl == 'grad':
						tst_dictionary_basis[grad_curl][loc][freq]['spectral_com'] = np.sum(w_grad * (np.abs(grad)**2 / np.sum(np.abs(grad)**2)).flatten())
					elif grad_curl == 'curl':
						tst_dictionary_basis[grad_curl][loc][freq]['spectral_com'] = np.sum(w_curl * (np.abs(curl)**2 / np.sum(np.abs(curl)**2)).flatten())
					else:
						raise ValueError('grad_curl must be grad or curl')

					# evaluate degree of spectral localization/leakage
					full_spec = np.vstack([grad, curl, harm]).flatten()
					tst_dictionary_basis[grad_curl][loc][freq]['spectral_precision'] = np.sum(np.abs(full_spec * freq_subset_binary)**2) / np.sum(np.abs(full_spec)**2)
					tst_dictionary_basis[grad_curl][loc][freq]['spectral_leakage_grad'] = np.sum(np.abs(grad)**2) / np.sum(np.abs(full_spec)**2)
					tst_dictionary_basis[grad_curl][loc][freq]['spectral_leakage_curl'] = np.sum(np.abs(curl)**2) / np.sum(np.abs(full_spec)**2)
					tst_dictionary_basis[grad_curl][loc][freq]['spectral_leakage_harm'] = np.sum(np.abs(harm)**2) / np.sum(np.abs(full_spec)**2)

					# degree of spatial localization (norm of D @ phi)
					tst_dictionary_basis[grad_curl][loc][freq]['spatial_localization'] = np.linalg.norm(D @ phi[:, -1])

					# degree of spectral localization (norm of S @ phi)
					tst_dictionary_basis[grad_curl][loc][freq]['spectral_localization'] = np.linalg.norm(S @ phi[:, -1])

		self.tst_basis = tst_dictionary_basis
		return self.tst_basis

	def topological_slepian_transform(self, f, tst_basis=None):
		"""
		:param f: numpy.ndarray
			E x T array, where E are the number of edges and T the number of time steps
		:param tst_dictionary_basis: dictionary
			topological slepian transform basis as returned by tst_dictionary_basis
		:return: topological slepian transform
		"""
		if tst_basis is None:
			tst_basis = self.tst_basis

		tst = {}
		for grad_curl in ['grad', 'curl']:
			tst[grad_curl] = {}
			for loc in tst_basis[grad_curl].keys():
				tst[grad_curl][loc] = {}
				for freq in tst_basis[grad_curl][loc].keys():
					tst[grad_curl][loc][freq] = np.dot(f.T, tst_basis[grad_curl][loc][freq]['basis'])

		return tst





