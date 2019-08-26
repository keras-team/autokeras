import tensorflow as tf
from tensorflow.python.util import nest

from autokeras.hypermodel import block
from autokeras.hypermodel import graph
from autokeras.hypermodel import hyperblock
from autokeras.hypermodel import node


def assemble(inputs, outputs, dataset):
    """Assemble the HyperBlocks based on the dataset and input output nodes.

    # Arguments
        inputs: A list of InputNode. The input nodes of the AutoModel.
        outputs: A list of HyperHead. The heads of the AutoModel.
        dataset: tf.data.Dataset. The training dataset.

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
            assemblers.append(ImageAssembler())
        if isinstance(input_node, node.StructuredDataInput):
            assemblers.append(StructuredDataAssembler())
        if isinstance(input_node, node.TimeSeriesInput):
            assemblers.append(TimeSeriesAssembler())

    # Iterate over the dataset to fit the assemblers.
    for x, _ in dataset:
        for temp_x, assembler in zip(x, assemblers):
            assembler.update(temp_x)

    # Assemble the model with assemblers.
    middle_nodes = []
    for input_node, assembler in zip(inputs, assemblers):
        middle_nodes.append(assembler.assemble(input_node))

    # Merge the middle nodes.
    if len(middle_nodes) > 1:
        output_node = block.Merge()(middle_nodes)
    else:
        output_node = middle_nodes[0]

    outputs = nest.flatten([output_blocks(output_node)
                            for output_blocks in outputs])
    return graph.GraphHyperModel(inputs, outputs)


class Assembler(object):
    """Base class for data type specific assemblers."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

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

    def assemble(self, input_node):
        # for image, use the num_instance to determine the range of the sizes of the
        # resnet and xception
        # use the image size to determine how the down sampling works, e.g. pooling.
        return hyperblock.ImageBlock()(input_node)


class StructuredDataAssembler(Assembler):
    pass


class TimeSeriesAssembler(Assembler):
    pass
