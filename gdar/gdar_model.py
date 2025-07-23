import numpy as np
from scipy import sparse, fftpack, signal
from .flow import FlowSignal

class GDARModel():
    """
    Implements the Graph Diffusion Autoregressive (GDAR) model along with other related multivariate autoregressive
    models. Furthermore, additional sparsity constraints can be applied to the model coefficients.

    For mathematical details on the model, see:
    [1] Schwock et al., *Inferring Neural Communication Dynamics from Field Potentials Using Graph Diffusion Autoregression*
    """

    def __init__(self, graph, K):
        """
        Initialize the GDAR model.

        Attributes:
            K (int): Order of the model.
            graph (:class:`gdar.graph.Graph`): The graph object on which the model operates.
            coeffs (np.ndarray or None): Coefficients of the model, initialized to None.
            _beta (np.ndarray or None): Vectorized coefficients of the model, initialized to None.
            sigma (np.ndarray or None): Noise covariance matrix, initialized to None.
            N (int): Number of nodes in the graph.
            E (int): Number of edges in the graph.

        Parameters:
            graph (:class:`gdar.graph.Graph`): The graph object on which the model operates.
            K (int): The order of the model.
        """
        self.K = K
        self.graph = graph
        self.coeffs = None  # N x N x K matrix where last dimension is indexed from t-K, ..., t-1
        self._beta = None  # vectorized version of coeffs
        self.sigma = None  # noise covariance matrix
        self.N = graph.N  # number of nodes
        self.E = graph.E  # number of edges

    def fit_gdar(self, data, I_sparse=None):
        """
        Fits the GDAR model to the data. The parameters are estimated in closed form using the generalized least squares
        (GLS) estimator.

        Parameters:
            data (np.ndarray): NxT data matrix, where T is the number of time steps and N is the number of nodes.
            I_sparse (np.ndarray or None): Additional enforced sparsity structure. Defined by horizontal stacking of
                adjacency matrix. If None, the sparsity structure is defined by the graph's adjacency matrix.

        Returns:
            np.ndarray: Coefficients of the fitted GDAR model.
        """
        coeffs = self.fit_restricted(data, sym=True, I_sparse=I_sparse)
        C = self.get_noise_covariance(data, coeffs)
        self.coeffs = self.fit_restricted(data, C=C, sym=True, I_sparse=I_sparse)
        return self.coeffs

    def fit_gvar(self, data, I_sparse=None):
        """
        Fits VAR model that is constrained to the graph structure. The parameters are estimated in closed form using the
        generalized least squares (GLS) estimator. This is the same as the GDAR model, but without the symmetry
        constraint. In [1], this is referred to as the eVAR model.

        Parameters:
            data (np.ndarray): NxT data matrix, where T is the number of time steps and N is the number of nodes.
            I_sparse (np.ndarray or None): Additional enforced sparsity structure. Defined by horizontal stacking of
                adjacency matrix. If None, the sparsity structure is defined by the graph's adjacency matrix.

        Returns:
            np.ndarray: Coefficients of the fitted GVAR model.
        """
        coeffs = self.fit_restricted(data, sym=False, I_sparse=I_sparse)
        C = self.get_noise_covariance(data, coeffs)
        self.coeffs = self.fit_restricted(data, C=C, sym=False, I_sparse=I_sparse)
        return self.coeffs

    def fit_var(self, data):
        """
        Fits standard VAR model without any constraints. Parameters are estimated in closed form using the ordinary
        least squares (OLS) estimator.

        Parameters:
            data (np.ndarray): NxT data matrix, where T is the number of time steps and N is the number of nodes.

        Returns:
            np.ndarray: Coefficients of the fitted VAR model. Shape is N x N x K, where K is the order of the model.
        """
        # compute model matrices
        _, y, _, Z = self._var_model_matrices(data)

        P = np.linalg.pinv(Z @ Z.T) @ Z
        self._beta = (y.reshape(self.N, Z.shape[1], order='F') @ P.T).reshape(-1,1, order='F')
        self.coeffs = self._beta_to_coeff(self._beta)
        return self.coeffs

    def fit_restricted(self, data, C=None, sym=False, I_sparse=None):
        """
        Fits VAR model with sparsity and symmetry constraints. If sym=True, this implemented the GDAR model for a given
        graph. Parameters are estimated in closed form using the generalized least squares (GLS) estimator.

        Parameters:
            data (np.ndarray) : NxT data matrix, where T is the number of time steps and N is the number of nodes.
            C (np.ndarray or None): Noise covariance matrix. If None, the noise covariance matrix is the identity matrix.
            sym (bool): If True, applies symmetry constraint to the model coefficients.
            I_sparse (np.ndarray or None): Additional enforced sparsity structure. Defined by horizontal stacking of
                adjacency matrix. If None, the sparsity structure is defined by the graph's adjacency matrix.

        Returns:
            np.ndarray: Coefficients of the fitted model. Shape is N x N x K, where K is the order of the model.
        """
        # initialize model matrices
        _, y, _, Z = self._var_model_matrices(data)
        R = self._sparsity_matrix(I_sparse)
        if sym:
            S = self._symmetry_matrix(I_sparse)
            R = R @ S
        if C is None:
            C = sparse.identity(self.N)
        else:
            C = np.linalg.pinv(C)

        # iterative implementation instead of kronecker product for memory and performance reasons
        Z2 = Z @ Z.T
        P = np.zeros((R.shape[0], R.shape[1]))
        for i, r in enumerate(R.T):
            P[:,i] = (C @ r.reshape(self.N, Z.shape[0], order='F') @ Z2.T).flatten(order='F')
        P = np.linalg.pinv(R.T @ P)

        b = (C @ y.reshape(self.N, Z.shape[1], order='F') @ Z.T).reshape(-1,1, order='F')
        b2 = R.T @ b
        beta = P @ b2
        self._beta = R @ beta
        self.coeffs = self._beta_to_coeff(self._beta)
        return self.coeffs

    def get_noise_covariance(self, data, coeffs=None):
        """
        Estimates noise covariance matrix from the residuals of the fitted model.

        Parameters:
            data (np.ndarray): NxT data matrix, where T is the number of time steps and N is the number of nodes.
            coeffs (np.ndarray or None): Coefficients of the fitted model. If None, uses the current model coefficients.

        Returns:
            np.ndarray: Estimated noise covariance matrix. Shape is N x N.
        """
        if coeffs is not None:
            self.coeffs = coeffs
        T, _, Y, Z = self._var_model_matrices(data)
        B_stacked = self.coeffs.reshape(self.N, -1, order='F')
        self.sigma = 1/T * (Y - B_stacked @ Z) @ (Y - B_stacked @ Z).T
        return self.sigma

    def _var_model_matrices(self, data):
        """
        Helper function that compute matrices for model fitting

        Parameters:
            data (np.ndarray): NxT data matrix, where T is the number of time steps and N is the number of nodes.

        Returns:
            T (int): Number of time steps.
            y (np.ndarray): Response variables, flattened.
            Y (np.ndarray): Response variables in matrix form.
            Z (np.ndarray): Regressor matrix.
        """
        T = data.shape[1] # number of time steps
        y = data[:, self.K:].flatten(order='F').reshape(-1,1)  # response variables
        Y = data[:, self.K:]  # response variables in matrix form
        Z = np.zeros((self.N * self.K, T - self.K))  # regressor matrix
        for i in range(T - self.K):
            Z[:,i] = data[:,i:i+self.K].flatten(order='F')

        return T, y, Y, Z

    def _beta_to_coeff(self, beta):
        """
        Helper function to reshape coefficient vector to coefficient matrix

        Parameters:
            beta (np.ndarray): Coefficient vector of shape (N^2 * K, 1) where N is the number of nodes and K is the order
                of the model.

        Returns:
            coeffs (np.ndarray): Coefficient matrix of shape (N, N, K) where N is the number of nodes and K is the order
                of the model.
        """
        coeffs = np.zeros((self.N, self.N, self.K))
        for k in range(self.K):
            for i in range(self.N):
                coeffs[:,i,k] = beta[k * self.N**2 + i * self.N:k * self.N**2 + (i+1) * self.N].flatten()
        return coeffs

    def _sparsity_matrix(self, I_spase=None):
        """
        Helper function to create sparsity constraint matrix from predefined graph

        Parameters:
            I_spase (np.ndarray or None): Additional enforced sparsity structure. Defined by horizontal stacking of
                adjacency matrix. If None, the sparsity structure is defined by the graph's adjacency matrix.
        Returns:
            R (sparse.csr_matrix): Sparsity constraint matrix of shape (N^2 * K, N_non_zero), where N is the number of
                nodes and K is the order of the model. N_non_zero is the number of non-zero entries in the sparsity
                structure.
        """
        if I_spase is None:
            I_spase = (self.graph.A == 1) * 1 + np.identity(self.N)
            I_spase = np.tile(I_spase, self.K)
        else:
            I_spase = I_spase
        c = I_spase.flatten(order='F')
        N_non_zero = int(np.sum(c))
        R = np.zeros((len(c), N_non_zero))
        cnt = 0
        for i in range(len(c)):
            if c[i] == 1:
                R[i, cnt] = 1.0
                cnt += 1
        R = sparse.csr_matrix(R)
        return R

    def _symmetry_matrix(self, I_sparse=None):
        """
        Helper function to create symmetry constraint matrix from predefined graph

        Parameters:
            I_sparse (np.ndarray or None): Additional enforced sparsity structure. Defined by horizontal stacking of
                adjacency matrix. If None, the sparsity structure is defined by the graph's adjacency matrix.
        Returns:
            S_full (sparse.csr_matrix): Symmetry constraint matrix.
        """
        if I_sparse is None:
            I_sparse = np.tile(self.graph.A + np.identity(self.N), self.K)
        else:
            I_sparse = I_sparse

        for k in range(self.K):
            total_params = int(np.sum(I_sparse[:,k*self.N:(k+1)*self.N]))
            all_param_to_edge_list = []
            unique_edge_to_param_list = {}
            cnt_unique_params = 0
            for i in range(self.N):
                for j in range(self.N):
                    if I_sparse[j,k*self.N + i] == 1.0:
                        all_param_to_edge_list.append((j,i))
                        if j >= i:
                            unique_edge_to_param_list[(j,i)] = cnt_unique_params
                            cnt_unique_params += 1
            S = np.zeros((total_params, total_params - (total_params - self.N) // 2))
            for i, e in enumerate(all_param_to_edge_list):
                ordered_e = e if e[0] >= e[1] else (e[1], e[0])
                S[i, unique_edge_to_param_list[ordered_e]] = 1
            if k == 0:
                S_full = S
            else:
                S_full = np.block([
                    [S_full, np.zeros((S_full.shape[0], S.shape[1]))],
                    [np.zeros((S.shape[0], S_full.shape[1])), S]
                ])
        S_full = sparse.csr_matrix(S_full)
        return S_full


    def spectrum(self, coeffs=None, n=1999):
        """
        Compute the spectrum of the model coefficients using the discrete Fourier transform (DFT). This is the basis for
        many functional connectivity metrics such as Directed Transfer Function (DTF) and Partial Directed Coherence
        (PDC).

        Parameters:
            coeffs (np.ndarray or None): Coefficients of the fitted model. If None, uses the current model coefficients.
            n (int): Number of frequency bins for the DFT.

        Returns:
            np.ndarray: Coefficient spectrum of shape (N, N, n), where N is the number of nodes and n is the number of
                frequency bins.
        """
        if coeffs is not None:
            self.coeffs = coeffs
        N = self.coeffs.shape[0]
        spec = fftpack.fft(np.dstack([np.eye(N), -self.coeffs[:,:,::-1]]), n=n)
        return spec

    def pdc(self, coeff_spec):
        """
        Compute Partial Directed Coherence (PDC) from coefficient spectrum. PDC is a measure of the directed
        connectivity between two nodes in a multivariate autoregressive model.

        Parameters:
            coeff_spec (np.ndarray): Coefficient spectrum of shape (N, N, n), where N is the number of nodes and n is
                the number of frequency bins.

        Returns:
            np.ndarray: PDC estimate of shape (N, N, n).
        """
        pdc = np.abs(coeff_spec / np.sqrt(np.sum(np.abs(coeff_spec)**2, axis=0)))
        return pdc

    def dtf(self, coeff_spec):
        """
        Compute Directed Transfer Function (DTF) from coefficient spectrum. DTF is a measure of the directed
        connectivity between two nodes in a multivariate autoregressive model.
        Parameters:
            coeff_spec (np.ndarray): Coefficient spectrum of shape (N, N, n), where N is the number of nodes and n is
                the number of frequency bins.

        Returns:
            np.ndarray: DTF estimate of shape (N, N, n).
        """
        spec_inv = np.zeros(coeff_spec.shape, dtype=np.complex_)
        for i in range(coeff_spec.shape[-1]):
            spec_inv[:,:,i] = np.linalg.inv(coeff_spec[:,:,i])
        dtf = np.abs(spec_inv / np.sqrt(np.sum(np.abs(spec_inv)**2, axis=1)))
        return dtf

    def param_connectivity(self, coeff_spec):
        """
        Compute parameter connectivity from coefficient spectrum. This is a measure of the strength of the connection
        between two nodes from the model coefficients and is defined as the absolute value of the coefficient spectrum
        normalized by the geometric mean of the denominators of the coefficients of the nodes.

        Parameters:
            coeff_spec (np.ndarray): Coefficient spectrum of shape (N, N, n), where N is the number of nodes and n is
                the number of frequency bins.

        Returns:
            np.ndarray: Parameter connectivity estimate of shape (N, N, n).
        """
        param_c = np.zeros(coeff_spec.shape)
        for e in self.graph.edge_list:
            denom1 = coeff_spec[e[0], e[0]]
            for i in range(self.N):
                if i != e[0]:
                    denom1 -= coeff_spec[e[0], i]
            denom2 = coeff_spec[e[1], e[1]]
            for i in range(self.N):
                if i != e[1]:
                    denom2 -= coeff_spec[e[1], i]
            c = np.abs(coeff_spec[e[0], e[1]]) / np.sqrt(np.abs(denom1) * np.abs(denom2))
            param_c[e[0], e[1]] = c
            param_c[e[1], e[0]] = c
        return param_c

    def get_flow_var(self, data, coeffs=None, constrained=False):
        """
        Compute the VAR flow signal (i.e. bidirectional flow). The computed flow matrix either has shape
        N x N x T - K + 1 if the flow is computed for all possible edges or shape 2 x E x T - K + 1 if the flow is
        computed only for the edges of the predefined graph. The flow is computed by convolving the data with the model
        coefficients.

        Parameters:
            data (np.ndarray): N x T data matrix, where N is the number of nodes and T is the number of time points.
            coeffs (np.ndarray): Fitted VAR model coefficients of appropriate shape.
            constrained (bool): If True, compute flow only along edges of the predefined graph.

        Returns:
            :class:`gdar.flow.FlowSignal`: FlowSignal object containing the computed flow signal.
        """
        if coeffs is not None:
            self.coeffs = coeffs
        N = self.coeffs.shape[0]
        if not constrained:
            flow = np.zeros((N, N, data.shape[1] - self.K + 1))
            for i in range(N):
                for j in range(N):
                    h = self.coeffs[i,j]
                    flow[i,j] = signal.convolve(data[i], h[::-1], mode='valid')
        else:
            E = self.graph.E
            flow = np.zeros((2, E, data.shape[1] - self.K + 1))
            for k, e in enumerate(self.graph.edge_list):
                h = self.coeffs[e[0], e[1]]
                flow[0, k, :] = signal.convolve(data[e[0]], h[::-1], mode='valid')

                h = self.coeffs[e[1], e[0]]
                flow[1, k, :] = signal.convolve(data[e[1]], h[::-1], mode='valid')

        flow_signal = FlowSignal(graph=self.graph, f=flow)
        return flow_signal

    def get_flow_gdar(self, data, coeffs=None):
        """
        Compute the GDAR flow signal (i.e. unidirectional flow) based on the model coefficients. The flow is computed by
        convolving the gradient of the data with the model coefficients. The flow matrix has shape E x T, where E is the
        number of edges in the graph and T is the number of time points minus K + 1 (the order of the model).

        Parameters:
            data (np.ndarray): NxT data matrix, where N is the number of nodes and T is the number of time points.
            coeffs (np.ndarray): Fitted GDAR model coefficients of appropriate shape.

        Returns:
            :class:`gdar.flow.FlowSignal`: FlowSignal object containing the computed flow signal.
        """
        if coeffs is not None:
            self.coeffs = coeffs
        N = self.coeffs.shape[0]
        E = self.graph.E
        flow = np.zeros((E, data.shape[1] - self.K + 1))
        for k, e in enumerate(self.graph.edge_list):
            h = self.coeffs[e[0], e[1]]
            data_grad = data[e[1]] - data[e[0]]
            flow[k, :] = signal.convolve(data_grad, h[::-1], mode='valid')

        flow_signal = FlowSignal(graph=self.graph, f=flow)
        return flow_signal



