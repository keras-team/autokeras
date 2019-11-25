"""This module compiles the graph at different stages of execution.

The information in one block may need to be shared to another block. The functions
in this block fetches the information for different blocks in the graph. It also
checks if the graph is valid, i.e., any blocks are connected incorrectly.
"""

import queue

from autokeras.hypermodel import base
from autokeras.hypermodel import block as block_module
from autokeras.hypermodel import head as head_module
from autokeras.hypermodel import hyperblock as hyperblock_module
from autokeras.hypermodel import node as node_module
from autokeras.hypermodel import preprocessor as preprocessor_module


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
        if isinstance(block, preprocessor_module.TextToIntSequence):
            embedding_block.max_features = block.max_features
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


def lightgbm_head(lightgbm_block):
    """Fetch the heads for LightGBMBlock."""
    heads = fetch_heads(lightgbm_block)
    if len(heads) > 1:
        raise ValueError('LightGBMBlock can only be connected to one head.')
    head = heads[0]
    if isinstance(head, head_module.ClassificationHead):
        classifier = preprocessor_module.LightGBMClassifier(seed=lightgbm_block.seed)
        classifier.num_classes = head.num_classes
        lightgbm_block.lightgbm_block = classifier
    if isinstance(head, head_module.RegressionHead):
        lightgbm_block.lightgbm_block = preprocessor_module.LightGBMRegressor(
            seed=lightgbm_block.seed)

    in_block = head
    # Check if the head has no other input but only LightGBMBlock.
    while in_block is not lightgbm_block:
        # The head has other inputs.
        if len(in_block.inputs) > 1:
            return
        in_block = in_block.inputs[0].in_blocks[0]
    head.identity = True


def feature_engineering_input(fe_block):
    """Fetch the column_types and column_names.

    The values are fetched for FeatureEngineering from StructuredDataInput.
    """
    if not isinstance(fe_block.inputs[0], node_module.StructuredDataInput):
        raise TypeError('FeatureEngineering block can only be used '
                        'with StructuredDataInput.')
    fe_block.column_types = fe_block.inputs[0].column_types
    fe_block.column_names = fe_block.inputs[0].column_names


def structured_data_block_heads(structured_data_block):
    structured_data_block.num_heads = len(fetch_heads(structured_data_block))


# Compile the graph before the preprocessing step.
BEFORE = {
    preprocessor_module.FeatureEngineering: feature_engineering_input,
    preprocessor_module.LightGBMBlock: lightgbm_head,
}

# Compile the graph after the preprocessing step.
AFTER = {
    block_module.EmbeddingBlock: embedding_max_features,
}

# Compile the HyperGraph.
HYPER = {**{
    hyperblock_module.StructuredDataBlock: structured_data_block_heads,
}, **BEFORE}
