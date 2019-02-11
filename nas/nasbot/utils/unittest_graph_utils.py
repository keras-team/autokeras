"""
  Test cases for functions in graph_utils.py
  -- kandasamy@cs.cmu.edu
"""

# pylint: disable=import-error
# pylint: disable=no-member
# pylint: disable=invalid-name
# pylint: disable=relative-import
# pylint: disable=no-name-in-module

import numpy as np
from ..utils import graph_utils
from time import clock
from ..utils.base_test_class import BaseTestClass, execute_tests
from scipy.sparse import dok_matrix


ERR_TOL_FRAC = 1e-5


class GraphUtilsTestCase(BaseTestClass):
  """ Unit test class for ancillary utilities. """

  def setUp(self):
    """ Sets up attributes. """
    self.num_vertices_values = [4, 10, 50, 100]

  def test_apsp_algos(self):
    """ Tests the all pairs shortest pair problems. """
    # Todo: write code to compute computation time.
    self.report('Testing All Pairs Shortest Paths for n=%s.'%(
                str(self.num_vertices_values)))
    for num_vert_val in self.num_vertices_values:
      # Floyd warshall
      A = graph_utils.construct_random_adjacency_matrix(num_vert_val, avg_edge_frac=0.4,
                                                        directed=True)
      num_edges = A.sum()
      fw_start_time = clock()
      fw_dist = graph_utils.apsp_floyd_warshall(A) # call floyd_warshall
      fw_end_time = clock()
      fw_time_taken = fw_end_time - fw_start_time
      # Dijkstra's
      dj_start_time = clock()
      dj_dist = graph_utils.apsp_dijkstra(A) # call dijkstra's
      dj_end_time = clock()
      dj_time_taken = dj_end_time - dj_start_time
      # Report times taken
      self.report('Time taken for n=%d, m=%d: FW: %0.4f, Dijkstra\'s: %0.4f.'%(
                  A.shape[0], num_edges, fw_time_taken, dj_time_taken))
      # Test if both are equal
      fw_infties = fw_dist == np.inf
      dj_infties = dj_dist == np.inf
      assert np.all(fw_infties == dj_infties)
      # Zer out the infinities and compare the norms
      fw_dist[fw_infties] = 0
      dj_dist[dj_infties] = 0
      error_tol = ERR_TOL_FRAC * np.linalg.norm(dj_dist + fw_dist)
      assert np.linalg.norm(fw_dist - dj_dist) < error_tol

  def test_kahn_topoligical_sort(self):
    """ Tests topological sorting. """
    self.report('Testing topological sort.')
    A = dok_matrix((6, 6))
    A[0, 2] = 1
    A[1, 2] = 1
    A[1, 3] = 1
    A[2, 3] = 1
    A[2, 4] = 1
    A[4, 3] = 1
    A[5, 0] = 1
    A[5, 1] = 1
    sorted_order, has_cycles = graph_utils.kahn_topological_sort(A, 5)
    assert has_cycles == False
    assert sorted_order == [5, 0, 1, 2, 4, 3] or sorted_order == [5, 1, 0, 2, 4, 3]
    # Now create a graph with cycles
    A[3, 0] = 1
    sorted_order, has_cycles = graph_utils.kahn_topological_sort(A, 5)
    assert has_cycles == True

if __name__ == '__main__':
  execute_tests()

