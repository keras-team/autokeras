import autokeras as ak
import os
import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = '5'

gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

data = ak.NERData()
data.loadData('../data/conll2000/train.txt')
data.printSentence(2)
data.printUniqTags()

word_index_offset = 2
tag_index_offset = 1

word_to_id = {w: (i + word_index_offset) for i, w in enumerate(data.vocab)}
word_to_id["UNK"] = 1
word_to_id["PAD"] = 0

id_to_word = {value: key for key, value in word_to_id.items()}

tag_to_id = {w: (i + tag_index_offset) for i, w in enumerate(data.uniq_tags)}
tag_to_id["PAD"] = 0

id_to_tag = {value: key for key, value in tag_to_id.items()}
