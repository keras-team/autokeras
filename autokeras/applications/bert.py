#https://www.tensorflow.org/official_models/fine_tuning_bert
# import os
#
# import numpy as np
# import matplotlib.pyplot as plt
#
# import tensorflow as tf
#
# import tensorflow_hub as hub
# import tensorflow_datasets as tfds
# tfds.disable_progress_bar()
#
# from official.modeling import tf_utils
# from official import nlp
# from official.nlp import bert
#
# # Load the required submodules
# import official.nlp.optimization
# import official.nlp.bert.bert_models
# import official.nlp.bert.configs
# import official.nlp.bert.run_classifier
# import official.nlp.bert.tokenization
# import official.nlp.data.classifier_data_lib
# import official.nlp.modeling.losses
# import official.nlp.modeling.models
# import official.nlp.modeling.networks
#
# gs_folder_bert = "gs://cloud-tpu-checkpoints/bert/keras_bert/uncased_L-12_H-768_A-12"
# tf.io.gfile.listdir(gs_folder_bert)
#
# # Set up tokenizer to generate Tensorflow dataset
# tokenizer = bert.tokenization.FullTokenizer(
#     vocab_file=os.path.join(gs_folder_bert, "vocab.txt"),
#      do_lower_case=True)
#
# print("Vocab size:", len(tokenizer.vocab))
#
#
# def encode_sentence(s, tokenizer):
#     tokens = list(tokenizer.tokenize(s))
#     tokens.append('[SEP]')
#     return tokenizer.convert_tokens_to_ids(tokens)
#
#
# def bert_encode(glue_dict, tokenizer):
#     num_examples = len(glue_dict["sentence1"])
#
#     sentence1 = tf.ragged.constant([
#         encode_sentence(s, tokenizer)
#         for s in np.array(glue_dict["sentence1"])])
#     sentence2 = tf.ragged.constant([
#         encode_sentence(s, tokenizer)
#         for s in np.array(glue_dict["sentence2"])])
#
#     cls = [tokenizer.convert_tokens_to_ids(['[CLS]'])] * sentence1.shape[0]
#     input_word_ids = tf.concat([cls, sentence1, sentence2], axis=-1)
#
#     input_mask = tf.ones_like(input_word_ids).to_tensor()
#
#     type_cls = tf.zeros_like(cls)
#     type_s1 = tf.zeros_like(sentence1)
#     type_s2 = tf.ones_like(sentence2)
#     input_type_ids = tf.concat(
#         [type_cls, type_s1, type_s2], axis=-1).to_tensor()
#
#     inputs = {
#         'input_word_ids': input_word_ids.to_tensor(),
#         'input_mask': input_mask,
#         'input_type_ids': input_type_ids}
#
#     return inputs
#
# # glue_train = bert_encode(glue['train'], tokenizer)
# # glue_train_labels = glue['train']['label']
# #
# # glue_validation = bert_encode(glue['validation'], tokenizer)
# # glue_validation_labels = glue['validation']['label']
# #
# # glue_test = bert_encode(glue['test'], tokenizer)
# # glue_test_labels  = glue['test']['label']
#
#
# # for key, value in glue_train.items():
# #   print(f'{key:15s} shape: {value.shape}')
# #
# # print(f'glue_train_labels shape: {glue_train_labels.shape}')
#
#
# ## BUILD THE MODEL
# import json
#
# bert_config_file = os.path.join(gs_folder_bert, "bert_config.json")
# config_dict = json.loads(tf.io.gfile.GFile(bert_config_file).read())
#
# bert_config = bert.configs.BertConfig.from_dict(config_dict)
#
# config_dict
# ## THE CALSSIFIER
#
# bert_classifier, bert_encoder = bert.bert_models.classifier_model(
#     bert_config, num_labels=2)
#
# print(tf.keras.utils.plot_model(bert_classifier, show_shapes=True, dpi=48))
#
#
# ## EXAMPLE
# # glue_batch = {key: val[:10] for key, val in glue_train.items()}
# #
# # bert_classifier(
# #     glue_batch, training=True
# # ).numpy()
#
# print(tf.keras.utils.plot_model(bert_encoder, show_shapes=True, dpi=48))
#
#
# #RESTORING THE ENCODER WEIGHTS
#
# checkpoint = tf.train.Checkpoint(model=bert_encoder)
# checkpoint.restore(
#     os.path.join(gs_folder_bert, 'bert_model.ckpt')).assert_consumed()
#
#
# #SET UP THE OPTIMIZER
#
# # Set up epochs and steps
# epochs = 3
# batch_size = 32
# eval_batch_size = 32
#
# train_data_size = len(glue_train_labels)
# steps_per_epoch = int(train_data_size / batch_size)
# num_train_steps = steps_per_epoch * epochs
# warmup_steps = int(epochs * train_data_size * 0.1 / batch_size)
#
# # creates an optimizer with learning rate schedule
# optimizer = nlp.optimization.create_optimizer(
#     2e-5, num_train_steps=num_train_steps, num_warmup_steps=warmup_steps)
#
# bert_classifier.fit(
#       glue_train, glue_train_labels,
#       validation_data=(glue_validation, glue_validation_labels),
#       batch_size=32,
#       epochs=epochs)
#
# #TEST EXAMPLE
#
# my_examples = bert_encode(
#     glue_dict = {
#         'sentence1':[
#             'The rain in Spain falls mainly on the plain.',
#             'Look I fine tuned BERT.'],
#         'sentence2':[
#             'It mostly rains on the flat lands of Spain.',
#             'Is it working? This does not match.']
#     },
#     tokenizer=tokenizer)
#
# result = bert_classifier(my_examples, training=False)
#
# result = tf.argmax(result).numpy()
# print(result)
#
# print(np.array(info.features['label'].names)[result])
# print(type(optimizer))
#
#
# # TRAIN THE MODEL
#
# metrics = [tf.keras.metrics.SparseCategoricalAccuracy('accuracy', dtype=tf.float32)]
# loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
#
# bert_classifier.compile(
#     optimizer=optimizer,
#     loss=loss,
#     metrics=metrics)

from typing import Optional

import tensorflow as tf
# from kerastuner.applications import resnet
# from kerastuner.applications import xception
# from tensorflow.keras import layers
from tensorflow.python.util import nest

# from autokeras.blocks import reduction
from autokeras.engine import block as block_module
# from autokeras.utils import layer_utils
# from autokeras.utils import utils
# from autokeras.basic import set_hp_value
import json
from autokeras.blocks import preprocessing
import os
# import tensorflow_hub as hub
from official.nlp import bert

# Load the required submodules
import official.nlp.optimization
import official.nlp.bert.bert_models
import official.nlp.bert.configs
import official.nlp.bert.run_classifier
import official.nlp.bert.tokenization
import official.nlp.data.classifier_data_lib
import official.nlp.modeling.losses
import official.nlp.modeling.models
import official.nlp.modeling.networks

gs_folder_bert = "gs://cloud-tpu-checkpoints/bert/keras_bert/uncased_L-12_H-768_A-12"
# tf.io.gfile.listdir(gs_folder_bert)
hub_url_bert = "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/2"

class BERT(block_module.Block):
    """Block for Pretrained BERT.

    # Arguments
        version: String. 'v1', 'v2' or 'next'. The type of ResNet to use.
            If left unspecified, it will be tuned automatically.
        pooling: String. 'avg', 'max'. The type of pooling layer to use.
            If left unspecified, it will be tuned automatically.
    """

    def __init__(self,
                 version: Optional[str] = None,
                 pooling: Optional[str] = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.version = version
        self.pooling = pooling

    def get_config(self):
        config = super().get_config()
        config.update({
            'version': self.version,
            'pooling': self.pooling})
        return config

    def build(self, hp, inputs=None):
        input_tensor = nest.flatten(inputs)[0]
        # input_shape = None

        # hp.Choice('version', ['v1', 'v2', 'next'], default='v2')
        # hp.Choice('pooling', ['avg', 'max'], default='avg')
        #
        # set_hp_value(hp, 'version', self.version)
        # set_hp_value(hp, 'pooling', self.pooling)
        #
        # model = super().build(hp)

        ## bert config file
        bert_config_file = os.path.join(gs_folder_bert, "bert_config.json")
        config_dict = json.loads(tf.io.gfile.GFile(bert_config_file).read())

        bert_config = bert.configs.BertConfig.from_dict(config_dict)

        bert_classifier, bert_encoder = bert.bert_models.classifier_model(
            bert_config, num_labels=2)

        ## TOKENIZER
        tokenizer = bert.tokenization.FullTokenizer(
            vocab_file=os.path.join(gs_folder_bert, "vocab.txt"),
            do_lower_case=True)

        output_node = preprocessing.TextVectorizationWithTokenizer(
            tokenizer=tokenizer).build(input_tensor)

        # hub_encoder = hub.KerasLayer(hub_url_bert, trainable=True)

        checkpoint = tf.train.Checkpoint(model=bert_encoder)

        checkpoint.restore(
            os.path.join(gs_folder_bert, 'bert_model.ckpt')).assert_consumed()

        output_node = bert_encoder(
            inputs=output_node,
            training=True,
        )

        return output_node