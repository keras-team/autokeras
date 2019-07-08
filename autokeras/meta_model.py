import statistics
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.python.util import nest

from autokeras.hypermodel import hyper_node, processor
from autokeras.hypermodel import hyper_block


def get_assembler(input_node):
    if isinstance(input_node, hyper_node.TextInput):
        return TextAssembler()
    if isinstance(input_node, hyper_node.ImageInput):
        return ImageAssembler()
    if isinstance(input_node, hyper_node.StructuredInput):
        return StructuredAssembler()
    if isinstance(input_node, hyper_node.TimeSeriesInput):
        return TimeSeriesAssembler()


def assemble(inputs, outputs, x, y):
    inputs = nest.flatten(inputs)
    outputs = nest.flatten(outputs)
    middle_nodes = []
    for input_node, temp_x in zip(inputs, x):
        assembler = get_assembler(input_node)
        middle_nodes.append(assembler.assemble(
            x=temp_x,
            input_node=input_node,
            y=y,
            outputs=outputs))

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

    def assemble(self, x, input_node, y, outputs):
        raise NotImplementedError


class TextAssembler(Assembler):

    def assemble(self, x, input_node, y, outputs):
        output_node = input_node
        sw_ratio = self.sw_ratio(x)
        if isinstance(input_node, hyper_node.TextNode):
            if sw_ratio < 1500:
                output_node = processor.TextToNgramVector()(output_node)
                output_node = hyper_block.DenseBlock()(output_node)
            else:
                output_node = processor.TextToSequenceVector()(output_node)
                output_node = hyper_block.EmbeddingBlock(
                    pretrained=(sw_ratio < 15000))(output_node)
                output_node = hyper_block.ConvBlock()(output_node)
        return output_node

    @staticmethod
    def sw_ratio(x):
        if isinstance(x, tf.data.Dataset):
            x = np.array(list(tfds.as_numpy(x))).astype(np.str)
        num_samples = len(x)
        tokenizer = tf.keras.preprocessing.text.Tokenizer()
        tokenizer.fit_on_texts(x)
        sequences = tokenizer.texts_to_sequences(x)
        num_words_per_sample = statistics.mean([len(seq) for seq in sequences])
        return num_samples / num_words_per_sample


class ImageAssembler(Assembler):
    def assemble(self, x, input_node, y, outputs):
        return hyper_block.ImageBlock()(input_node)


class StructuredAssembler(Assembler):
    def assemble(self, x, input_node, y, outputs):
        pass


class TimeSeriesAssembler(Assembler):
    def assemble(self, x, input_node, y, outputs):
        pass
