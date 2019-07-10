import statistics
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.python.util import nest

from autokeras import utils
from autokeras.hypermodel import hyper_node
from autokeras.hypermodel import hyper_head
from autokeras.hypermodel import hyper_block
from autokeras.hypermodel import processor


def assemble(inputs, outputs, dataset):
    inputs = nest.flatten(inputs)
    outputs = nest.flatten(outputs)
    assemblers = []
    for input_node in inputs:
        if isinstance(input_node, hyper_node.TextInput):
            assemblers.append(TextAssembler())
        if isinstance(input_node, hyper_node.ImageInput):
            assemblers.append(ImageAssembler())
        if isinstance(input_node, hyper_node.StructuredInput):
            assemblers.append(StructuredAssembler())
        if isinstance(input_node, hyper_node.TimeSeriesInput):
            assemblers.append(TimeSeriesAssembler())

    # Iterate over the dataset to fit the assemblers.
    for x, y in dataset:
        for temp_x, assembler in zip(x, assemblers):
            assembler.update(temp_x)

    middle_nodes = []
    for input_node, assembler in zip(inputs, assemblers):
        middle_nodes.append(assembler.assemble(input_node))

    if len(middle_nodes) > 1:
        output_node = hyper_block.Merge()(middle_nodes)
    else:
        output_node = middle_nodes[0]

    return nest.flatten([output_blocks(output_node)
                         for output_blocks in outputs])
    # all inputs, all train_x y, all heads
    # for text data we just follow the rules on that page.
    # for image, use the num_instance to determine the range of the sizes of the
    # resnet and xception
    # use the image size to determine how the down sampling works, e.g. pooling.


class Assembler(object):
    def __init__(self, **kwargs):
        super(Assembler, self).__init__(**kwargs)

    def update(self, x):
        pass


class TextAssembler(Assembler):
    def __init__(self, **kwargs):
        super(TextAssembler, self).__init__(**kwargs)
        self._num_words = 0
        self._num_samples = 0

    def update(self, x):
        """Update the assembler sample by sample.

        Args:
            x: tf.Tensor. A data instance from input dataset.
        """
        x = x.numpy().decode('utf-8')
        self._num_samples += 1
        tokenizer = tf.keras.preprocessing.text.Tokenizer()
        tokenizer.fit_on_texts([x])
        self._num_words += len(tokenizer.texts_to_sequences([x])[0])

    def sw_ratio(self):
        return self._num_samples * self._num_samples / self._num_words

    def assemble(self, input_node):
        """Assemble the HyperBlocks for text input.

        Implemented according to Google Developer, Machine Learning Guide, Text
        Classification, Step 2.5: Choose a Model.

        Args:
            input_node: HyperNode. The input node for the AutoModel.

        Returns:
            A HyperNode. The output node of the assembled model.
        """
        output_node = input_node
        ratio = self.sw_ratio()
        if isinstance(input_node, hyper_node.TextNode):
            if ratio < 1500:
                output_node = processor.TextToNgramVector()(output_node)
                output_node = hyper_block.DenseBlock()(output_node)
            else:
                output_node = processor.TextToSequenceVector()(output_node)
                output_node = hyper_block.EmbeddingBlock(
                    pretrained=(ratio < 15000))(output_node)
                output_node = hyper_block.ConvBlock(separable=True)(output_node)
        return output_node


class ImageAssembler(Assembler):
    pass


class StructuredAssembler(Assembler):
    pass


class TimeSeriesAssembler(Assembler):
    pass
