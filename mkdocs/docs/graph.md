##NetworkDescriptor
###__init__
###n_dense
###n_conv
###add_conv_width
###add_dense_width
###add_skip_connection
##Graph
A class represent the neural architecture graph of a Keras model.
Graph extracts the neural architecture graph from a Keras model. Each node in the graph is a intermediate tensor between layers. Each layer is an edge in the graph.  Notably, multiple edges may refer to the same layer. (e.g. WeightedAdd layer is adding two tensor into one tensor. So it is related to two edges.)
####Attributes
**model**: The Keras model, from which to extract the graph.

**node_list**: A list of tensors, the indices of the list are the identifiers.

**layer_list**: A list of Keras layers, the indices of the list are the identifiers.

**node_to_id**: A dict instance mapping from tensors to their identifiers.

**layer_to_id**: A dict instance mapping from Keras layers to their identifiers.

**layer_id_to_input_node_ids**: A dict instance mapping from layer identifiers
    to their input nodes identifiers.

**adj_list**: A two dimensional list. The adjacency list of the graph. The first dimension is
    identified by tensor identifiers. In each edge list, the elements are two-element tuples
    of (tensor identifier, layer identifier).

**reverse_adj_list**: A reverse adjacent list in the same format as adj_list.

**middle_layer_vis**: A boolean list marking whether a node has been visited or not.

###__init__
###clear_operation_history
###n_nodes
Return the number of nodes in the model.

###n_layers
Return the number of layers in the model.

###_add_node
Add node to node list if it not in node list.

###_add_new_node
###_add_edge
Add edge to the graph.

###_redirect_edge
Redirect the edge to a new node. Change the edge originally from u_id to v_id into an edge from u_id to new_v_id while keeping all other property of the edge the same.

###_replace_layer
Replace the layer with a new layer.

###_topological_order
Return the topological order of the node ids.

###_get_pooling_layers
###_depth_first_search
###_search
Search downward the graph for widening the layers.

####Args
**u**: The starting node identifier.

**start_dim**: The dimension to insert the additional dimensions.

**total_dim**: The total number of dimensions the layer has before widening.

**n_add**: The number of dimensions to add.

###_upper_layer_width
###to_conv_deeper_model
Insert a convolution, batch-normalization, relu block after the target block.

####Args
**target_id**: A convolutional layer ID. The new block should be inserted after the relu layer
    in its conv-batch-relu block.

**kernel_size**: An integer. The kernel size of the new convolutional layer.

####Returns
A new Keras model with the inserted block.
###to_wider_model
Widen the last dimension of the output of the pre_layer.

####Args
**pre_layer_id**: A convolutional layer or dense layer.

**n_add**: The number of dimensions to add.

####Returns
A new Keras model with the widened layers.
###to_dense_deeper_model
Insert a dense layer after the target layer.

####Args
**target_id**: A dense layer.

####Returns
A new Keras model with an inserted dense layer.
###_insert_new_layers
###_block_end_node
###_dense_block_end_node
###_conv_block_end_node
Args: layer_id: the convolutional layer ID.

####Returns
The input node ID of the last layer in the convolutional block.
###to_add_skip_model
Add a weighted add skip connection from before start node to end node.

####Args
**start_id**: The convolutional layer ID, after which to start the skip-connection.

**end_id**: The convolutional layer ID, after which to end the skip-connection.

####Returns
A new Keras model with the added connection.
###to_concat_skip_model
Add a weighted add concatenate connection from before start node to end node.

####Returns
A new Keras model with the added connection.
###extract_descriptor
###produce_model
Build a new Keras model based on the current graph.

###_layer_ids_in_order
###_layer_ids_by_type
###_conv_layer_ids_in_order
###_dense_layer_ids_in_order
###deep_layer_ids
###wide_layer_ids
###skip_connection_layer_ids
