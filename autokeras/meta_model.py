import tensorflow as tf
from kerastuner.engine import hyperparameters as hp_module
from tensorflow.python.util import nest
import numpy as np

from autokeras.hypermodel import block
from autokeras.hypermodel import graph
from autokeras.hypermodel import head
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
            assemblers.append(StructuredDataAssembler())
        if isinstance(input_node, node.TimeSeriesInput):
            assemblers.append(TimeSeriesAssembler())

    # Iterate over the dataset to fit the assemblers.
    hps = []
    for x, y in dataset:
        for temp_x, assembler in zip(x, assemblers):
            assembler.update(temp_x)
            hps += assembler.hps

    # Single structured data classification or regression.
    if (len(inputs) == 1 and len(outputs) == 1 and
            isinstance(inputs[0], node.StructuredDataInput)):
        assemblers[0].infer_column_types()
        if isinstance(outputs[0], head.ClassificationHead):
            output_block = hyperblock.StructuredDataClassifierBlock(
                column_types=assemblers[0].column_types,
                head=outputs[0])
        else:
            output_block = hyperblock.StructuredDataRegressorBlock(
                column_types=assemblers[0].column_types,
                head=outputs[0])
        return graph.GraphHyperModel(inputs, output_block(inputs))

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
    hm = graph.GraphHyperModel(inputs, outputs)
    hm.set_hps(hps)
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
        return block(input_node)


class StructuredDataAssembler(Assembler):
    """Assembler for structured data. which infers the column types for the data.

    A column will be judged as categorical if the number of different values is less
    than 5% of the number of instances.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.column_types = []
        self.count_nan = None
        self.count_numerical = None
        self.count_categorical = None
        self.count_unique_numerical = []
        self.num_col = None

    def update(self, x):
        # calculate the statistics.
        x = nest.flatten(x)[0].numpy()
        if self.num_col is None:
            self.num_col = len(x)
            self.count_nan = np.zeros(self.num_col)
            self.count_numerical = np.zeros(self.num_col)
            self.count_categorical = np.zeros(self.num_col)
            for i in range(len(x)):
                self.count_unique_numerical.append({})
        for i in range(self.num_col):
            x[i] = x[i].decode('utf-8')
            if x[i] == 'nan':
                self.count_nan[i] += 1
            elif x[i] == 'True':
                self.count_categorical[i] += 1
            elif x[i] == 'False':
                self.count_categorical[i] += 1
            else:
                try:
                    tmp_num = float(x[i])
                    self.count_numerical[i] += 1
                    if tmp_num not in self.count_unique_numerical[i]:
                        self.count_unique_numerical[i][tmp_num] = 1
                    else:
                        self.count_unique_numerical[i][tmp_num] += 1
                except ValueError:
                    self.count_categorical[i] += 1

    def infer_column_types(self):
        for i in range(self.num_col):
            if self.count_categorical[i] > 0:
                self.column_types.append('categorical')
            elif len(self.count_unique_numerical[i])/self.count_numerical[i] < 0.05:
                self.column_types.append('categorical')
            else:
                self.column_types.append('numerical')

    def assemble(self, input_node):
        self.infer_column_types()
        return hyperblock.StructuredDataBlock(self.column_types)(input_node)


class TimeSeriesAssembler(Assembler):
    pass
