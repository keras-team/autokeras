import tensorflow as tf
from kerastuner.engine import hyperparameters as hp_module
from tensorflow.python.util import nest

from autokeras.hypermodel import block as block_module
from autokeras.hypermodel import graph
from autokeras.hypermodel import hyperblock
from autokeras.hypermodel import node


def assemble(inputs, outputs, dataset, seed=None):
    """Assemble the HyperBlocks based on the dataset and input output nodes.

    # Arguments
        inputs: A list of InputNode. The input nodes of the AutoModel.
        outputs: A list of HyperHead. The heads of the AutoModel.
        dataset: tf.data.Dataset. The training dataset.
        seed: Int. Random seed.

    # Returns
        A list of HyperNode. The output nodes of the AutoModel.
    """
    inputs = nest.flatten(inputs)
    outputs = nest.flatten(outputs)

    assemblers = []
    for input_node in inputs:
        if isinstance(input_node, node.TextInput):
            assemblers.append(TextAssembler())
        if isinstance(input_node, node.ImageInput):
            assemblers.append(ImageAssembler(seed=seed))
        if isinstance(input_node, node.StructuredDataInput):
            assemblers.append(StructuredDataAssembler(seed=seed))
        if isinstance(input_node, node.TimeSeriesInput):
            assemblers.append(TimeSeriesAssembler())
    # Iterate over the dataset to fit the assemblers.
    hps = []
    for x, _ in dataset:
        for temp_x, assembler in zip(x, assemblers):
            assembler.update(temp_x)
            hps += assembler.hps

    # Assemble the model with assemblers.
    middle_nodes = []
    for input_node, assembler in zip(inputs, assemblers):
        middle_nodes.append(assembler.assemble(input_node))

    # Merge the middle nodes.
    if len(middle_nodes) > 1:
        output_node = block_module.Merge()(middle_nodes)
    else:
        output_node = middle_nodes[0]

    outputs = nest.flatten([output_blocks(output_node)
                            for output_blocks in outputs])
    hm = graph.HyperGraph(inputs=inputs, outputs=outputs, override_hps=hps)
    return hm


class Assembler(object):
    """Base class for data type specific assemblers."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.hps = []

    def update(self, x):
        """Update the assembler sample by sample.

        # Arguments
            x: tf.Tensor. A data instance from input dataset.
        """
        pass

    def assemble(self, input_node):
        """Assemble the HyperBlocks for text input.

        # Arguments
            input_node: HyperNode. The input node for the AutoModel.

        # Returns
            A HyperNode. The output node of the assembled model.
        """
        raise NotImplementedError


class TextAssembler(Assembler):
    """Selecting the HyperBlocks for text tasks based the training data.

    Implemented according to Google Developer, Machine Learning Guide,
    Text Classification, Step 2.5: Choose a Model.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._num_words = 0
        self._num_samples = 0

    def update(self, x):
        x = x.numpy().decode('utf-8')
        self._num_samples += 1
        tokenizer = tf.keras.preprocessing.text.Tokenizer()
        tokenizer.fit_on_texts([x])
        self._num_words += len(tokenizer.texts_to_sequences([x])[0])

    def sw_ratio(self):
        return self._num_samples * self._num_samples / self._num_words

    def assemble(self, input_node):
        ratio = self.sw_ratio()
        if not isinstance(input_node, node.TextNode):
            raise ValueError('The input_node should be a TextNode.')
        pretraining = 'random'
        if ratio < 1500:
            vectorizer = 'ngram'
        else:
            vectorizer = 'sequence'
            if ratio < 15000:
                pretraining = 'glove'

        return hyperblock.TextBlock(vectorizer=vectorizer,
                                    pretraining=pretraining)(input_node)


class ImageAssembler(Assembler):
    """Assembles the ImageBlock based on training dataset."""

    def __init__(self, seed=None, **kwargs):
        super().__init__(**kwargs)
        self.seed = seed
        self._shape = None
        self._num_samples = 0

    def update(self, x):
        self._shape = x.shape
        self._num_samples += 1

    def assemble(self, input_node):
        block = hyperblock.ImageBlock(seed=self.seed)
        if max(self._shape[0], self._shape[1]) < 32:
            if self._num_samples < 10000:
                self.hps.append(hp_module.Choice(
                                block.name + '_resnet/v1/conv4_depth', [6],
                                default=6))
                self.hps.append(hp_module.Choice(
                                block.name + '_resnet/v2/conv4_depth', [6],
                                default=6))
                self.hps.append(hp_module.Choice(
                                block.name + '_resnet/next/conv4_depth', [6],
                                default=6))
                self.hps.append(hp_module.Int(
                                block.name + '_xception/num_residual_blocks', 2, 4,
                                default=4))
        output_node = block(input_node)
        output_node = block_module.SpatialReduction()(output_node)
        return block_module.DenseBlock()(output_node)


class StructuredDataAssembler(Assembler):
    """Assembler for structured data. which infers the column types for the data.

    # Arguments
        seed: Int. Random seed.
    """

    def __init__(self, seed=None, **kwargs):
        super().__init__(**kwargs)
        self.seed = seed

    def assemble(self, input_node):
        return hyperblock.StructuredDataBlock(seed=self.seed)(input_node)


class TimeSeriesAssembler(Assembler):
    pass
