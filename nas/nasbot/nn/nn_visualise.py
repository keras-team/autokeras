"""
  Harness for visualising a neural network.
  -- kandasamy@cs.cmu.edu
"""

# pylint: disable=invalid-name

import functools
import graphviz as gv
import os

# Parameters for plotting
_SAVE_FORMAT = 'eps'
# _SAVE_FORMAT = 'png'
_LAYER_SHAPE = 'rectangle'
_IPOP_SHAPE = 'circle'
_LAYER_FONT = 'DejaVuSans'
_IPOP_FONT = 'Helvetica'
_LAYER_FONTSIZE = '16'
_FILLCOLOR = 'transparent'
_IPOP_FONTSIZE = '12'
_IPOP_FILLCOLOR = '#ffc0cb'
_DECISION_FILLCOLOR = '#98fb98'
_GRAPH_STYLES = {
  'graph': {
    'fontsize': _LAYER_FONTSIZE,
    'rankdir': 'TB',
    'label': None,
  },
  'nodes': {
  },
  'edges': {
    'arrowhead': 'open',
    'fontsize': '12',
  }
}

GV_GRAPH = functools.partial(gv.Graph, format=_SAVE_FORMAT)
GV_DIGRAPH = functools.partial(gv.Digraph, format=_SAVE_FORMAT)

# Utilities for adding nodes, edges and styles -------------------------------------------
def add_nodes(graph, nodes):
  """ Adds nodes to the graph. """
  for n in nodes:
    if isinstance(n, tuple):
      graph.node(n[0], **n[1])
    else:
      graph.node(n)
  return graph

def add_edges(graph, edges):
  """ Adds edges to the graph. """
  # pylint: disable=star-args
  for e in edges:
    if isinstance(e[0], tuple):
      graph.edge(*e[0], **e[1])
    else:
      graph.edge(*e)
  return graph

def apply_styles(graph, styles):
  """ Applies styles to the graph. """
  graph.graph_attr.update(
      ('graph' in styles and styles['graph']) or {}
  )
  graph.node_attr.update(
      ('nodes' in styles and styles['nodes']) or {}
  )
  graph.edge_attr.update(
      ('edges' in styles and styles['edges']) or {}
  )
  return graph

# Wrappers for tedious routines ----------------------------------------------------------
def _get_ip_layer(layer_idx):
  """ Returns a tuple representing the input layer. """
  return (str(layer_idx), {'label': 'i/p', 'shape': 'circle', 'style': 'filled',
                           'fillcolor': _IPOP_FILLCOLOR, 'fontsize': _IPOP_FONTSIZE,
                           'fontname': _IPOP_FONT})

def _get_op_layer(layer_idx):
  """ Returns a tuple representing the output layer. """
  return (str(layer_idx), {'label': 'o/p', 'shape': 'circle', 'style': 'filled',
                           'fillcolor':  _IPOP_FILLCOLOR, 'fontsize': _IPOP_FONTSIZE,
                           'fontname': _IPOP_FONT})

def _get_layer(layer_idx, nn, for_pres):
  """ Returns a tuple representing the layer label. """
  if nn.layer_labels[layer_idx] in ['ip', 'op']:
    fill_colour = _IPOP_FILLCOLOR
  elif nn.layer_labels[layer_idx] in ['softmax', 'linear']:
    fill_colour = _DECISION_FILLCOLOR
  else:
    fill_colour = _FILLCOLOR
  label = nn.get_layer_descr(layer_idx, for_pres)
  return (str(layer_idx), {'label': label, 'shape': 'rectangle', 'fillcolor': fill_colour,
                           'style': 'filled', 'fontname': _LAYER_FONT})

def _get_edge(layer_idx_start, layer_idx_end):
  """ Returns a tuple which is an edge. """
  return (str(layer_idx_start), str(layer_idx_end))

def _get_edges(conn_mat):
  """ Returns all edges. """
  starts, ends = conn_mat.nonzero()
  return [_get_edge(starts[i], ends[i]) for i in range(len(starts))]

# Main API ------------------------------------------------------------------------------
def visualise_nn(nn, save_file_prefix, fig_label=None, for_pres=True):
  """ The main API which will be used to visualise the network. """
  # First create nodes in the order
  nodes = [_get_layer(i, nn, for_pres) for i in range(nn.num_layers)]
  edges = _get_edges(nn.conn_mat)
  nn_graph = GV_DIGRAPH()
  add_nodes(nn_graph, nodes)
  add_edges(nn_graph, edges)
  graph_styles = _GRAPH_STYLES
  graph_styles['graph']['label'] = fig_label
  apply_styles(nn_graph, graph_styles)
  nn_graph.render(save_file_prefix)
  if os.path.exists(save_file_prefix):
    # graphviz also creates another file in the name of the prefix. delete it.
    os.remove(save_file_prefix)

def visualise_list_of_nns(list_of_nns, save_dir, fig_labels=None, fig_file_names=None,
                          for_pres=False):
  """ Visualises a list of neural networks. """
  if fig_labels is None:
    fig_labels = [None] * len(list_of_nns)
  if fig_file_names is None:
    fig_file_names = [str(idx) for idx in range(len(list_of_nns))]
  for idx, nn in enumerate(list_of_nns):
    save_file_prefix = os.path.join(save_dir, fig_file_names[idx])
    visualise_nn(nn, save_file_prefix, fig_labels[idx], for_pres)

