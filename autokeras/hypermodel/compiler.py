"""This module compiles the graph at different stages of execution.

The information in one block may need to be shared to another block. The functions
in this block fetches the information for different blocks in the graph. It also
checks if the graph is valid, i.e., any blocks are connected incorrectly.
"""

import queue

import tensorflow as tf

from autokeras.hypermodel import base
from autokeras.hypermodel import block as block_module
from autokeras.hypermodel import head as head_module
from autokeras.hypermodel import node as node_module


def embedding_max_features(embedding_block):
    """Fetch the max_features value for embedding block from TextToIntSequence."""
    if embedding_block.max_features:
        return
    input_node = embedding_block.inputs[0]
    while True:
        if not input_node.in_blocks:
            raise ValueError('If Embedding block is not using with '
                             'TextToIntSequence, max_features must be '
                             'specified.')
        block = input_node.in_blocks[0]
        if isinstance(block, block_module.TextToIntSequence):
            embedding_block.max_features = block.max_tokens
            return
        input_node = block.inputs[0]


def fetch_heads(source_block):
    """Get the downstream head blocks for a given block in the network.

    # Arguments
        source_block: Block. The source block for the search for heads.

    # Returns
        A list of Head instances.
    """
    heads = []
    visited_blocks = set()
    visited_blocks.add(source_block)
    q = queue.Queue()
    q.put(source_block)
    while not q.empty():
        block = q.get()
        if isinstance(block, base.Head):
            heads.append(block)
        for output_node in block.outputs:
            for next_block in output_node.out_blocks:
                if next_block not in visited_blocks:
                    visited_blocks.add(next_block)
                    q.put(next_block)
    return heads


def feature_encoding_input(fe_block):
    """Fetch the column_types and column_names.

    The values are fetched for FeatureEncoding from StructuredDataInput.
    """
    if not isinstance(fe_block.inputs[0], node_module.StructuredDataInput):
        raise TypeError('FeatureEncoding block can only be used '
                        'with StructuredDataInput.')
    fe_block.column_types = fe_block.inputs[0].column_types
    fe_block.column_names = fe_block.inputs[0].column_names


def structured_data_block_heads(structured_data_block):
    structured_data_block.num_heads = len(fetch_heads(structured_data_block))


# Compile the graph.
COMPILE_FUNCTIONS = {
    block_module.Embedding: embedding_max_features,
    block_module.StructuredDataBlock: structured_data_block_heads,
    block_module.FeatureEncoding: feature_encoding_input,
}

ALL_CLASSES = {
    **vars(base),
    **vars(node_module),
    **vars(head_module),
    **vars(block_module),
}


def serialize(obj):
    return tf.keras.utils.serialize_keras_object(obj)


def deserialize(config, custom_objects=None):
    return tf.keras.utils.deserialize_keras_object(
        config,
        module_objects={**ALL_CLASSES},
        custom_objects=custom_objects,
        printable_module_name='graph')
