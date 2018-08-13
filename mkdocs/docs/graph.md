##Graph
A class representing the neural architecture graph of a Keras model.
Graph extracts the neural architecture graph from a Keras model. Each node in the graph is a intermediate tensor between layers. Each layer is an edge in the graph. Notably, multiple edges may refer to the same layer. (e.g. Add layer is adding two tensor into one tensor. So it is related to two edges.)
#####Attributes
* **weighted**: A boolean of whether the weights and biases in the neural network
    should be included in the graph.

* **input_shape**: A tuple of integers, which does not include the batch axis.

* **node_list**: A list of integers. The indices of the list are the identifiers.

* **layer_list**: A list of stub layers. The indices of the list are the identifiers.

* **node_to_id**: A dict instance mapping from node integers to their identifiers.

* **layer_to_id**: A dict instance mapping from stub layers to their identifiers.

* **layer_id_to_input_node_ids**: A dict instance mapping from layer identifiers
    to their input nodes identifiers.

* **adj_list**: A two dimensional list. The adjacency list of the graph. The first dimension is
    identified by tensor identifiers. In each edge list, the elements are two-element tuples
    of (tensor identifier, layer identifier).

* **reverse_adj_list**: A reverse adjacent list in the same format as adj_list.

* **operation_history**: A list saving all the network morphism operations.

* **vis**: A dictionary of temporary storage for whether an local operation has been done
    during the network morphism.

###n_nodes
Return the number of nodes in the model.

###n_layers
Return the number of layers in the model.

###_add_node
Add node to node list if it is not in node list.

###_add_edge
Add an edge to the graph.

###_redirect_edge
Redirect the edge to a new node. Change the edge originally from `u_id` to `v_id` into an edge from `u_id` to `new_v_id` while keeping all other property of the edge the same.

###_replace_layer
Replace the layer with a new layer.

###topological_order
Return the topological order of the node ids.

###_search
Search the graph for widening the layers.

#####Args
* **u**: The starting node identifier.

* **start_dim**: The position to insert the additional dimensions.

* **total_dim**: The total number of dimensions the layer has before widening.

* **n_add**: The number of dimensions to add.

###to_conv_deeper_model
Insert a relu-conv-bn block after the target block.

#####Args
* **target_id**: A convolutional layer ID. The new block should be inserted after the block.

* **kernel_size**: An integer. The kernel size of the new convolutional layer.

###to_wider_model
Widen the last dimension of the output of the pre_layer.

#####Args
* **pre_layer_id**: The ID of a convolutional layer or dense layer.

* **n_add**: The number of dimensions to add.

###to_dense_deeper_model
Insert a dense layer after the target layer.

#####Args
* **target_id**: The ID of a dense layer.

###_conv_block_end_node
Get the input node ID of the last layer in the block by layer ID. Return the input node ID of the last layer in the convolutional block.

#####Args
* **layer_id**: the convolutional layer ID.

###to_add_skip_model
Add a weighted add skip-connection from after start node to end node.

#####Args
* **start_id**: The convolutional layer ID, after which to start the skip-connection.

* **end_id**: The convolutional layer ID, after which to end the skip-connection.

###to_concat_skip_model
Add a weighted add concatenate connection from after start node to end node.

#####Args
* **start_id**: The convolutional layer ID, after which to start the skip-connection.

* **end_id**: The convolutional layer ID, after which to end the skip-connection.

###produce_model
Build a new model based on the current graph.

###produce_keras_model
Build a new keras model based on the current graph.

