"""
  Some utilities for graph operations.
  -- kandasamy@cs.cmu.edu
"""

# pylint: disable=import-error
# pylint: disable=no-member
# pylint: disable=invalid-name
# pylint: disable=relative-import


from copy import deepcopy
import numpy as np
from scipy.linalg import eigh as sp_eigh
# Local imports
from nas.nasbot.utils.general_utils import get_nonzero_indices_in_vector


def construct_random_adjacency_matrix(num_vertices, avg_edge_frac=0.4, directed=False):
  """ Constructs a random adjacency matrix where avg_edge_frac of the edges are connected
      on average. """
  M = np.random.random((num_vertices, num_vertices))
  threshold = avg_edge_frac if directed else 2 * avg_edge_frac
  M = M if directed else (M + M.T)
  A = (M < threshold).astype(np.float)
  np.fill_diagonal(A, 0)
  return A


def get_undirected_adj_mat_from_directed(directed_adj_mat):
  """ Given a directed adjacency matrix, returns the undirected matrix. """
  num_nodes = directed_adj_mat.shape[0]
  ret = directed_adj_mat + directed_adj_mat.T
  ret[np.diag_indices(num_nodes)] = directed_adj_mat[np.diag_indices(num_nodes)]
  return ret


def is_valid_adj_mat(A):
  """ Returns true if A is a valid adjacency matrix. i.e. it is symmetric and all
      diagonals are 0. """
  return (A == A.T).all() and (A.diagonal() == 0).all()


def get_normalised_laplacian(A):
  """ Returns the (Symmetric) Normalised Graph Laplacian from an adjacency matrix A. """
  assert is_valid_adj_mat(A)
  with np.errstate(divide='ignore'):
    D_sqrt_inv = 1./ np.sqrt(A.sum(axis=0))
    D_sqrt_inv[D_sqrt_inv == np.inf] = 0
  L = np.eye(A.shape[0]) - (A * D_sqrt_inv) * D_sqrt_inv[:, np.newaxis]
  return L


def get_smallest_symm_eig(M, smallest_eigval_idx=None):
  """ Returns the smallest_eigval_idx eigenvalues and eigenvectors of the symmetric
      matrix M. """
  smallest_eigval_idx = (M.shape[0] - 1) if smallest_eigval_idx is None \
                        else smallest_eigval_idx
  eig_val_indices = [0, min(smallest_eigval_idx, M.shape[0]-1)]
  eigvals, eigvecs = sp_eigh(M, eigvals=eig_val_indices)
  return eigvals, eigvecs


def compute_lap_embedding_from_laplacian(L, num_proj_dimensions=None,
                                         proj_to_sphere=True):
  """ Returns the representation using Laplacian eigenvectors and in the unit cube. """
  num_proj_dimensions = num_proj_dimensions if num_proj_dimensions is not None \
                        else L.shape[0] # return entire matrix.
  _, eigvecs = get_smallest_symm_eig(L, num_proj_dimensions-1)
  if proj_to_sphere:  # each row has unit euclidean norm
    row_wise_norms = np.sqrt(np.sum(eigvecs**2, axis=1))
    sphere_repr = eigvecs / row_wise_norms[:, np.newaxis]
  # Pad with zeros
  if sphere_repr.shape[1] < num_proj_dimensions:
    zero_mat = np.zeros((L.shape[0], num_proj_dimensions - sphere_repr.shape[1]))
    sphere_repr = np.concatenate((sphere_repr, zero_mat), axis=1)
  return sphere_repr


def compute_lap_embedding_from_graph(A, num_proj_dimensions=None, proj_to_sphere=True):
  """ Returns the representation using Laplacian eigenvectors and in the unit cube.
      Computes this from the adjacency matrix by first computing the Laplacian."""
  L = get_normalised_laplacian(A)
  return compute_lap_embedding_from_laplacian(L, num_proj_dimensions, proj_to_sphere)


def apsp_floyd_warshall_costs(A):
  """ Runs the Floyd Warshall algorithm to return an nxn matrix which computes the
      all pairs shortest paths. Here A(i,j) is treated as the distance from i to j.
      So 0's will be counted as 0's. Non-edges should be specified as infinities.
      Just copying the pseudo code in Wikipedia."""
  dist = deepcopy(A)
  np.fill_diagonal(dist, 0)
  n = A.shape[0]
  for k in range(n):
    for i in range(n):
      for j in range(n):
        if dist[i, j] > dist[i, k] + dist[k, j]:
          dist[i, j] = dist[i, k] + dist[k, j]
  return dist


def apsp_floyd_warshall(A):
  """ Runs the Floyd Warshall algorithm to return an nxn matrix which computes the
      all pairs shortest paths. Here 0's denote non-edges. """
  dist = deepcopy(A)
  # Create an initial matrix with infinity for all non-edges and zero on the diagonals.
  dist[dist == 0] = np.inf
  np.fill_diagonal(dist, 0)
  return apsp_floyd_warshall_costs(dist)


def get_children(node_idx, A):
  """ Returns the children of node_idx according to the adjacency matrix A. """
  return get_nonzero_indices_in_vector(A[node_idx])

def get_parents(node_idx, A):
  """ Returns the parents of node_idx according to the adjacency matrix A. """
  return get_nonzero_indices_in_vector(A[:, node_idx])


def apsp_dijkstra(A):
  """ Runs Dijkstra's on all nodes to compute the all pairs shortest paths. """
  vertex_dists = []
  for vertex in range(A.shape[0]):
    curr_vertex_dists = dijkstra(A, vertex)
    vertex_dists.append(curr_vertex_dists)
  return np.array(vertex_dists)


def dijkstra(A, source, non_edges_are_zero_or_inf='zero'):
  """ Run's dijkstra's on the vertex to produce the shortest path to all nodes.
      Just copyng the pseudo code in Wikipedia.
      non_edges_are_zero_or_inf indicate whether a non-edge is indicated as a 0 or
      inf in A.
  """
  vertex_is_remaining = np.array([1] * A.shape[0])
  all_vertices = np.array(range(A.shape[0]))
  all_dists = np.array([np.inf] * A.shape[0])
  all_dists[source] = 0
  while sum(vertex_is_remaining) > 0:
    # Find the minimum and remove it.
    rem_dists = deepcopy(all_dists)
    rem_dists[vertex_is_remaining == 0] = np.nan
    u = np.nanargmin(rem_dists)
    vertex_is_remaining[u] = 0
    if np.all(np.logical_not(np.isfinite(rem_dists))):
      break
    # Now apply dijkstra's updates
    if non_edges_are_zero_or_inf == 'zero':
      u_nbd = all_vertices[A[u] > 0]
    elif non_edges_are_zero_or_inf == 'inf':
      u_nbd = all_vertices[A[u] < np.inf]
    else:
      raise ValueError('non_edges_are_zero_or_inf should be \'zero\' or \'inf\'.')
    for v in u_nbd:
      alt = all_dists[u] + A[u][v]
      if alt < all_dists[v]:
        all_dists[v] = alt
  return all_dists


def compute_nn_path_lengths(A, top_order, path_type):
  """ Computes the path lengths on a NN with adjacency matrix A to top_order[-1].
      top_order is a topological ordering of the nodes.
      A(i,j) is finite means there is an edge between i and j.
  """
  if path_type in 'shortest':
    get_curr_length_from_child_lengths = min
  elif path_type == 'longest':
    get_curr_length_from_child_lengths = max
  elif path_type == 'rw':
    get_curr_length_from_child_lengths = lambda x: sum(x) / float(len(x))
  else:
    raise ValueError('Unknown path_type: %s.'%(path_type))
  # Now the algorithm
  all_vertices = np.array(range(A.shape[0]))
  source = top_order[-1]
  all_dists = np.array([np.inf] * A.shape[0])
  all_dists[source] = 0
  for layer_idx in reversed(top_order[:-1]):
    curr_children = all_vertices[A[layer_idx] < np.inf]
    children_path_lengths = [all_dists[ch_idx] for ch_idx in curr_children]
    children_path_lengths = [children_path_lengths[i] + A[layer_idx, curr_children[i]] for
                             i in range(len(curr_children))]
    all_dists[layer_idx] = get_curr_length_from_child_lengths(children_path_lengths)
  return all_dists


def kahn_topological_sort(A, start_nodes):
  """ Applies Kahn's algorithm to return a topological sort of the graph. Starts with
      start_nodes. A is adjacency graph. Following pseudo code in Wikipedia. """
  ret = []
  edges = deepcopy(A)
  S = deepcopy(start_nodes) if hasattr(start_nodes, '__iter__') else [start_nodes]
  while len(S) > 0:
    curr_node = S.pop(0)
    ret.append(curr_node)
    curr_nbd = get_nonzero_indices_in_vector(edges[curr_node])
    edges[curr_node] = 0
    for neighbor in curr_nbd:
      if not edges[:, neighbor].sum() > 0:
        S.append(neighbor)
  has_cycles = False if edges.sum() == 0 else True
  return ret, has_cycles

