import os

BERT_DIR = "gs://cloud-tpu-checkpoints/bert/keras_bert/uncased_L-12_H-768_A-12"
BERT_CONFIG_PATH = os.path.join(BERT_DIR, "bert_config.json")
BERT_CHECKPOINT_PATH = os.path.join(BERT_DIR, "bert_model.ckpt")
BERT_VOCAB_PATH = os.path.join(BERT_DIR, "vocab.txt")
