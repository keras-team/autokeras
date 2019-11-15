
class Constant(object):
    # Text
    VOCABULARY_SIZE = 20000
    # Download link for word embedding
    EMDEDDING_PRETRAINING = {
        # Refer to https://fasttext.cc/docs/en/english-vectors.html
        'fasttext': {'URL': 'https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M.vec.zip',
                     'EMBEDDING_DIM': 300,
                     'FILE_NAME': 'wiki-news-300d-1M.vec',
                     'HAS_HEADER': True,
                     'EXTRACT': True
        },
        # Refer to https://nlp.stanford.edu/projects/glove/
        'glove': {'URL': 'http://nlp.stanford.edu/data/glove.6B.zip',
                  'EMBEDDING_DIM': 100,
                  'FILE_NAME': 'glove.6B.100d.txt',
                  'HAS_HEADER': False,
                  'EXTRACT': True
        },
        # Refer to https://code.google.com/archive/p/word2vec/
        # Pretrained text file refer to http://vectors.nlpl.eu/repository/
        'word2vec': {'URL': 'http://vectors.nlpl.eu/repository/11/1.zip',
                     'EMBEDDING_DIM': 300,
                     'FILE_NAME': 'model.txt',
                     'HAS_HEADER': True,
                     'EXTRACT': True
        }
    }
