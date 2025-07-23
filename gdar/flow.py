import numpy as np
from scipy import linalg
from copy import deepcopy

class FlowSignal():
	"""
	Class defining flow signal as well as useful transforms for decomposing flow signal into various spatial modes. The
	basis for the decomposition is the Hodge decomposition of the graph Laplacian, which decomposes the flow signal into
	gradient, curl, and harmonic components.

	For mathematical details on the Hodge decomposition, see:
	[2] S. Barbarossa and S. Sardellitti, "Topological Signal Processing Over Simplicial Complexes," in IEEE
	Transactions on Signal Processing
	"""
	def __init__(self, graph, f=None):
		"""
		Attributes:
			graph (:class:`gdar.graph.Graph`): The graph object on which the model operates.
			f (numpy.ndarray): Flow signal defined on the edges of the graph, shape (E, T) where E is the number of edges
				and T is the number of time steps. If None, no flow signal is set.
			w_grad (numpy.ndarray): Eigenvalues of the gradient spectrum, shape (N-1,).
			V_grad (numpy.ndarray): Eigenvectors of the gradient spectrum, shape (E, N-1).
			w_curl (numpy.ndarray): Eigenvalues of the curl spectrum shape (M,). The size M is determined by the number
			of non-zero eigenvalues of the upper Hodge Laplacian B2 @ B2.T.
			V_curl (numpy.ndarray): Eigenvectors of the curl spectrum, shape (E, M), where M is the number of non-zero
				eigenvalues of the upper Hodge Laplacian B2 @ B2.T.
			w_harm (numpy.ndarray): Eigenvalues of the harmonic spectrum, shape (K,). The size K is determined by the
				number of zero eigenvalues of the Hodge Laplacian B1 @ B1.T + B2 @ B2.T.
			V_harm (numpy.ndarray): Eigenvectors of the harmonic spectrum, shape (E, K), where K is the number of zero
				eigenvalues of the Hodge Laplacian B1 @ B1.T + B2 @ B2.T.

		Parameters:
			graph (:class:`gdar.graph.Graph`): The graph object on which the model operates.
			f (numpy.ndarray, optional): Flow signal defined on the edges of the graph, shape (E, T) where E is the number
				of edges and T is the number of time steps. If None, no flow signal is set.
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
		"""
		Set the flow signal for the FlowSignal object.

		Parameters:
			f (numpy.ndarray): Flow signal defined on the edges of the graph, shape (E, T) where E is the number
				of edges and T is the number of time steps.
		"""
		self.f = f

	def set_graph(self, graph):
		"""
		Set the graph for the FlowSignal object.

		Parameters:
			graph (:class:`gdar.graph.Graph`): The graph object on which the model operates.
		"""
		self.graph = graph

	def hodge_decomposition(self, mode='full'):
		"""
		Compute the Hodge decomposition of the flow signal on the graph. This defines the gradient, curl, and harmonic
		eigenfunctions of the flow signal and serves as the basis for the Flow Fourier transform.

		Parameters:
			mode (str): The mode of decomposition. Can be 'grad', 'curl', or 'full'. If 'grad', only gradient modes are
				computed. If 'curl', only curl modes are computed. If 'full', both gradient and curl modes are computed,
				as well as harmonic modes. Note that to compute the gradient componenet the node to edge
				incidence matrix B1 and to compute the curl comonent the edge to triangle incidence matrix B2 must be
				defined in the graph.

		Returns:
			tuple: Depending on the mode, returns the following:
				- If mode is 'grad': (w_grad, V_grad)
				- If mode is 'curl': (w_curl, V_curl)
				- If mode is 'full': (w_grad, V_grad, w_curl, V_curl, w_harm, V_harm)

		"""
		if self.graph.B1 is not None and mode == 'grad':
			w_grad, V_grad = np.linalg.eigh(self.graph.B1 @ self.graph.B1.T)
			self.w_grad = w_grad[1:] # skip zero eigenvalue
			self.V_grad = self.graph.B1.T @ V_grad[:, 1:]

			return self.w_grad, self.V_grad,

		if self.graph.B2 is not None and mode == 'curl':
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

	def flow_ft(self, f=None, idx=None, spectral_analysis=True, mode='full'):
		"""
		Compute the Flow Fourier Transform of the flow signal on the graph. This decomposes the flow signal into
		gradient, curl, and harmonic components based on the Hodge decomposition of the graph Laplacian. Note that the
		eigenvectors of the gradient and curl components are not unit norm. To get a meaningful spectrum, the
		eigenvectors are normalized to have unit norm first if spectral_analysis is True. The hodge decomposition needs
		to be computed first by calling the hodge_decomposition.

		Parameters:
			f (numpy.ndarray, optional): Flow signal defined on the edges of the graph, shape (E, T) where E is the number
				of edges and T is the number of time steps. If None, use self.f.
			idx (int or array-like, optional): Time indices for which the Flow Fourier Transform is computed. If None,
				compute the Flow Fourier Transform for all time steps.
			spectral_analysis (bool): If True, normalize the eigenvectors of the gradient and curl components to have
				unit norm. This is useful for spectral analysis of the flow signal.
			mode (str): The mode of decomposition. Can be 'grad', 'curl', or 'full'. If 'grad', only the gradient
				spectrum is computed. If 'curl', only the curl spectrum is computed. If 'full', both gradient
				and curl spectra are computed, as well as harmonic spectrum.

		Returns:
		"""
		if f is None:
			f = self.f
		if idx is None:
			idx = np.arange(f.shape[1])

		if mode not in ['grad', 'curl', 'full']:
			raise ValueError("mode must be 'grad', 'curl', or 'full'")

		# gradient spectrum
		if mode == 'grad' or mode == 'full':
			if spectral_analysis:
				# normalize V_grad so that all spatial eigenvectors have unit norm
				V_grad_norm = np.linalg.norm(self.V_grad, axis=0)
				V_grad = self.V_grad / V_grad_norm[np.newaxis, :]
			else:
				V_grad = self.V_grad
			F_grad = V_grad.T @ f[:,idx]

		# curl spectrum
		if mode == 'curl' or mode == 'full':
			if spectral_analysis:
				# normalize V_curl so that all spatial eigenvectors have unit norm
				V_curl_norm = np.linalg.norm(self.V_curl, axis=0)
				V_curl = self.V_curl / V_curl_norm[np.newaxis, :]
			else:
				V_curl = self.V_curl
			F_curl = V_curl.T @ f[:,idx]

		# harmonic spectrum
		if mode == 'full':
			if spectral_analysis:
				# normalize V_harm so that all spatial eigenvectors have unit norm
				V_harm_norm = np.linalg.norm(self.V_harm, axis=0)
				V_harm = self.V_harm / V_harm_norm[np.newaxis, :]
			else:
				V_harm = self.V_harm
			F_harm = V_harm.T @ f[:,idx]


		if mode == 'grad':
			return self.w_grad, F_grad
		elif mode == 'curl':
			return self.w_curl, F_curl
		elif mode == 'full':
			return self.w_grad, F_grad, self.w_curl, F_curl, self.w_harm, F_harm
		else:
			raise ValueError("mode must be 'grad', 'curl', or 'full'")

	def spectrogram(self, f=None, avg_win=1, mode='full'):
		"""
		Compute the spectrogram of the flow signal on the graph. This computes the magnitude square of the gradient
		and curl spectra of the flow signal and averages them over a sliding window of size avg_win. The hodge
		decomposition needs to be computed first by calling the function `hodge_decomposition`.

		Parameters:
			f (numpy.ndarray, optional): Flow signal defined on the edges of the graph, shape (E, T) where E is the number
				of edges and T is the number of time steps. If None, use self.f.
			avg_win (int): Size of the sliding window for averaging the spectra.
			mode (str): The mode of decomposition. Can be 'grad', 'curl', or 'full'. If 'grad', only the gradient
				spectrogram is computed. If 'curl', only the curl spectrogram is computed. If 'full', both gradient
				and curl spectrograms are computed.

		Returns:
			tuple: Depending on the mode, returns the following:
				- If mode is 'grad': (w_grad, Spec_grad)
				- If mode is 'curl': (w_curl, Spec_curl)
				- If mode is 'full': (w_grad, Spec_grad, w_curl, Spec_curl)
		"""
		if f is None:
			f = self.f

		if mode not in ['grad', 'curl', 'full']:
			raise ValueError("mode must be 'grad', 'curl', or 'full'")

		if mode == 'grad' or mode == 'full':
			w_grad, F_grad = self.flow_ft(f=f, spectral_analysis=True, mode='grad')
			if F_grad is not None:
				Spec_grad = np.zeros((len(w_grad), int(np.ceil(f.shape[1] / avg_win))))
				F_grad_square = F_grad**2
				for i in range(int(f.shape[1] / avg_win)):
					Spec_grad[:,i] = np.mean(F_grad_square[:,i*avg_win:(i+1)*avg_win], axis=1)

		if mode == 'curl' or mode == 'full':
			w_curl, F_curl = self.flow_ft(f=f, spectral_analysis=True, mode='curl')
			if F_curl is not None:
				Spec_curl = np.zeros((len(w_curl), int(np.ceil(f.shape[1] / avg_win))))
				F_curl_square = F_curl**2
				for i in range(int(f.shape[1] / avg_win)):
					Spec_curl[:,i] = np.mean(F_curl_square[:,i*avg_win:(i+1)*avg_win], axis=1)

		if mode == 'grad':
			return w_grad, Spec_grad
		elif mode == 'curl':
			return w_curl, Spec_curl
		elif mode == 'full':
			return w_grad, Spec_grad, w_curl, Spec_curl

	def get_flow_adjacency_matrix(self, index=0, f=None):
		"""
		Convert the flow signal into an adjacency matrix representation. This is useful for computing measures of
		network communication such as communicability.

		Parameters:
			index (int): Index of the flow signal if f is a matrix. If f is None, use self.f.
			f (numpy.ndarray, optional): Flow signal or matrix. If None, use self.f.

		Returns:
			numpy.ndarray: Flow signal represented as an adjacency matrix of shape (N, N), where N is the number of
			nodes in the graph. The flow signal is represented as a directed graph. That is, the adjacency matrix is
			asymmetric.
		"""
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