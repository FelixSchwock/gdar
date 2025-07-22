import numpy as np
from scipy import sparse, fftpack, signal
from .graph import Graph
from .flow import FlowSignal
import cvxpy as cp
from tqdm.notebook import tqdm

        
def grid_graph(node_positions):
    """
    Creates a grid graph

    :param node_positions: dictionary with node positions. Each entry should be of the form node_id: np.array(x, y)
    :return: graph object
    """
    edge_list = []
    graph = Graph()

    for v1 in node_positions:
        pos_v1 = node_positions[v1]

        left_neighbor = [None, np.inf]
        bottom_neighbor = [None, np.inf]

        for v2 in node_positions:
            pos_v2 = node_positions[v2]
            dist = pos_v1 - pos_v2
            if dist[1] == 0 and dist[0] > 0 and dist[0] < left_neighbor[1]:
                left_neighbor[0] = v2
                left_neighbor[1] = dist[0]

            elif dist[0] == 0 and dist[1] > 0 and dist[1] < bottom_neighbor[1]:
                bottom_neighbor[0] = v2
                bottom_neighbor[1] = dist[1]

        if left_neighbor[0] is not None:
            edge_list.append((np.min([v1, left_neighbor[0]]), np.max([v1, left_neighbor[0]]), {'weight': 1}))
        if bottom_neighbor[0] is not None:
            edge_list.append((np.min([v1, bottom_neighbor[0]]), np.max([v1, bottom_neighbor[0]]), {'weight': 1}))

    graph.generate_from_edge_list(edge_list)
    return graph

def nn_graph(node_positions, n_neighbors=8):
    """
    Creates a nearest neighbor graph

    :param node_positions: dictionary with node positions. Each entry should be of the form node_id: np.array(x, y)
    :param n_neighbors: number of neighbors to connect
    :return: graph object
    """
    edge_list = []
    graph = Graph()

    for v1 in node_positions:
        pos_v1 = node_positions[v1]

        dist_to_v1 = []
        v2_list = []
        for v2 in node_positions:
            pos_v2 = node_positions[v2]
            dist = np.sum((pos_v1 - pos_v2)**2)
            dist_to_v1.append(dist)
            v2_list.append(v2)
        idx_sort = np.argsort(dist_to_v1)
        v2_list = np.array(v2_list)
        v1_neighbors = v2_list[idx_sort[1:n_neighbors+1]]

        for v2 in v1_neighbors:
            if (np.min([v1, v2]), np.max([v1, v2]), {'weight': 1}) not in edge_list:
                edge_list.append((np.min([v1, v2]), np.max([v1, v2]), {'weight': 1}))

    graph.generate_from_edge_list(edge_list)
    return graph

def proximity_graph(node_positions, dist_th=5):
    """
    Creates graph based on node proximity. All nodes that are less or equal
    than dist_th away from each other are connected

    :param node_positions: dictionary with node positions. Each entry should be of the form node_id: np.array(x, y)
    :param dist_th: distance threshold
    :return: graph object
    """
    edge_list = []
    graph = Graph()
    for v1 in node_positions:
        pos_v1 = node_positions[v1]

        v2_list = []
        for v2 in node_positions:
            pos_v2 = node_positions[v2]
            dist = np.sqrt(np.sum((pos_v1 - pos_v2)**2))
            if dist <= dist_th and dist != 0:
                v2_list.append(v2)

        for v2 in v2_list:
            if (np.min([v1, v2]), np.max([v1, v2]), {'weight': 1}) not in edge_list:
                edge_list.append((np.min([v1, v2]), np.max([v1, v2]), {'weight': 1}))

    graph.generate_from_edge_list(edge_list)
    return graph

def custom_proximity_graph(node_positions, dist=None):
    """
    Creates graph based on node proximity, where the distance threshold can be specified for each node individually

    :param node_positions: dictionary with node positions. Each entry should be of the form node_id: np.array(x, y)
    :param dist: distance threshold. Can be a tuple (dist1, dist2) or a dictionary with the form
    dist_dct = {
        dist1 : [d1_n1, d1_n2, ...],
        dist2 : [d2_n1, d2_n2, ...],
        ...
    }
    If None, the dist dictionary is initialized such that for our ECoG arrays with variable spacing (Opti Stim and Reach
    datasets), an 8-neighbor nearest neighbor graph is achieved both the denser and sparser portion of the array.
    """
    graph = Graph()

    init_dist_dct = False
    if dist is None:
        dist_dct = {
            1.5 : [],
            2.9 : []
        }
        dist = (1.5, 2.9)
        init_dist_dct = True
    elif type(dist) is tuple:
        dist_dct = {
            dist[0]: [],
            dist[1]: []
        }
        init_dist_dct = True
    else:
        dist_dct = dist

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
                if (np.min([v1, v2]), np.max([v1, v2]), {'weight': 1}) not in edge_list:
                    edge_list.append((np.min([v1, v2]), np.max([v1, v2]), {'weight': 1}))

    graph.generate_from_edge_list(edge_list)
    return graph

class CVARModel():
    """
    Implements Multivariate autoregressive model with constraints for diffusion and sparsity.

    """
    def __init__(self, graph, K):
        """
        :param graph: graph object
        :param K: order of the model
        """
        self.K = K
        self.graph = graph
        self.coeffs = None  # N x N x K matrix where last dimension is indexed from t-K, ..., t-1
        self._beta = None  # vectorized version of coeffs
        self.sigma = None  # noise covariance matrix
        self.N = graph.N  # number of nodes
        self.E = graph.E  # number of edges

    def fit_unrestricted(self, data):
        """
        Fits VAR model without any constraints
        :param data: NxT data matrix
        :return: coefficients
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
        graph
        :param data: NxT data matrix
        :param C: covariance matrix
        :param sym: if True assume symmatrix coefficient matrix
        :param I_sparse: additional enforced sparsity structure. Defined by
            horizontal stacking of adjacency matrix
        :return: coefficients
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

    def fit_restricted_cvxpy(self, data, C=None, sym=False, l1_param=0.0, l2_param=0.0, I_sparse=None):
        """
        Same as fit_restricted, but using cvxpy to solve the optimization problem. This is slower and seems to be less
        accurate for large datasets, however it is more flexible and can be used if additional sparsity penalties are
        desired.
        :param data: NxT data matrix
        :param C: covariance matrix
        :param sym: if True assume symmatrix coefficient matrix
        :param l1_param: penalty for l1 norm
        :param l2_param: penalty for l2 norm
        :param I_sparse: additional enforced sparsity structure. Defined by
            horizontal stacking of adjacency matrix
        :return: coefficients
        """
        # solve OLS optimization
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
        P = R.T @ P

        b = (C @ y.reshape(self.N, Z.shape[1], order='F') @ Z.T).reshape(-1,1, order='F')
        b2 = b.T @ R

        # use cvxpy to solve problem with constraint
        P_cp = cp.atoms.affine.wraps.psd_wrap(P)
        beta = cp.Variable((self.N + self.E) * self.K)
        #cost = 0.5 * cp.quad_form(beta, P_cp) - b2 @ beta + l1_param * cp.norm1(beta) + l2_param * cp.pnorm(beta, p=2)**2
        cost = 0.5 * cp.quad_form(beta, P_cp) - b2 @ beta + l2_param * cp.pnorm(beta, p=2)**2
        prob = cp.Problem(cp.Minimize(cost))
        #prob.solve(solver='OSQP', eps_abs=1e-9, eps_rel=1e-9, max_iter=100000)
        #prob.solve(solver='ECOS', abstol=1e-9, reltol=1e-9, max_iters=100000)
        prob.solve()

        self._beta = R @ beta.value
        self.coeffs = self._beta_to_coeff(self._beta)
        return self.coeffs

    def get_noise_covariance(self, data, coeffs=None):
        """
        Estimates noise covariance matrix
        :param data: NxT data matrix
        :param coeffs: coefficients
        :return: noise covariance matrix
        """
        if coeffs is not None:
            self.coeffs = coeffs
        T, _, Y, Z = self._var_model_matrices(data)
        B_stacked = self.coeffs.reshape(self.N, -1, order='F')
        self.sigma = 1/T * (Y - B_stacked @ Z) @ (Y - B_stacked @ Z).T
        return self.sigma

    def _var_model_matrices(self, data):
        """
        Compute matrices for model fitting
        :param data: NxT data matrix
        :return: number of time steps, response variables, response variables in matrix form, regressor matrix
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
        Reshape coefficient vector to coefficient matrix
        :param beta: coefficient vector
        :return: coefficient matrix
        """
        coeffs = np.zeros((self.N, self.N, self.K))
        for k in range(self.K):
            for i in range(self.N):
                coeffs[:,i,k] = beta[k * self.N**2 + i * self.N:k * self.N**2 + (i+1) * self.N].flatten()
        return coeffs

    def _sparsity_matrix(self, I_spase=None):
        """
        Create sparsity constraint matrix from predefined graph
        :param I_spase: additional enforced sparsity structure.
        :return: sparsity constraint matrix
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
        Create symmetry constraint matrix from predefined graph
        :param I_sparse: sparsity structure.
        :return: symmetry constraint matrix
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
        Compute the spectrum of the model coefficients
        :param coeffs: coefficients
        :param n: number of frequency points
        :return: DFT spectrum
        """
        if coeffs is not None:
            self.coeffs = coeffs
        N = self.coeffs.shape[0]
        spec = fftpack.fft(np.dstack([np.eye(N), -self.coeffs[:,:,::-1]]), n=n)
        return spec

    def pdc(self, coeff_spec):
        """
        Compute partial directed coherence from coefficient spectrum
        :param coeff_spec: coefficient spectrum
        :return: partial directed coherence
        """
        pdc = np.abs(coeff_spec / np.sqrt(np.sum(np.abs(coeff_spec)**2, axis=0)))
        return pdc

    def dtf(self, coeff_spec):
        """
        Compute directed transfer function from coefficient spectrum
        :param coeff_spec: coefficient spectrum
        :return: directed transfer function
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
        :param coeff_spec: coefficient spectrum
        :return: parameter connectivity
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

    def get_flow_mvar(self, data, coeffs=None, constrained=False):
        """
        Compute the VAR flow signal (i.e. bidirectional flow)
        :param data: NxT data matrix
        :param coeffs: coefficients
        :param constrained: if True, compute flow only for edges of the predefined graph
        :return: flow signal
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

    def get_flow_diffusion(self, data, coeffs=None):
        """
        Compute the GDAR flow signal
        :param data: NxT data matrix
        :param coeffs: coefficients
        :return: flow signal
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



