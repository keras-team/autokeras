from collections import namedtuple

GoogleDriveFile = namedtuple('GoogleDriveFile', ['google_drive_id', 'local_name'])


class Constant:
    BACKEND = 'torch'
    # Data
    VALIDATION_SET_SIZE = 0.08333
    CUTOUT_HOLES = 1
    CUTOUT_RATIO = 0.5

    # Searcher
    MAX_MODEL_NUM = 1000
    BETA = 2.576
    KERNEL_LAMBDA = 1.0
    T_MIN = 0.0001
    N_NEIGHBOURS = 8
    MAX_MODEL_SIZE = (1 << 25)
    MAX_LAYER_WIDTH = 4096
    MAX_LAYERS = 200

    # Grid Dimensions
    LENGTH_DIM = 0
    WIDTH_DIM = 1

    # Default Search Space
    DEFAULT_LENGTH_SEARCH = [50, 75, 100]
    DEFAULT_WIDTH_SEARCH = [64, 128, 256]

    # Model Defaults

    DENSE_DROPOUT_RATE = 0.5
    CONV_DROPOUT_RATE = 0.25
    MLP_DROPOUT_RATE = 0.25
    CONV_BLOCK_DISTANCE = 2
    DENSE_BLOCK_DISTANCE = 1
    MODEL_LEN = 3
    MLP_MODEL_LEN = 3
    MLP_MODEL_WIDTH = 5
    MODEL_WIDTH = 64
    POOLING_KERNEL_SIZE = 2

    # ModelTrainer

    DATA_AUGMENTATION = True
    MAX_ITER_NUM = 200
    MIN_LOSS_DEC = 1e-4
    MAX_NO_IMPROVEMENT_NUM = 5
    MAX_BATCH_SIZE = 128
    LIMIT_MEMORY = False
    SEARCH_MAX_ITER = 200

    # Text Classifier

    BERT_TRAINER_EPOCHS = 4
    BERT_TRAINER_BATCH_SIZE = 32

    # text preprocessor

    EMBEDDING_DIM = 100
    MAX_SEQUENCE_LENGTH = 400
    MAX_NB_WORDS = 5000
    EXTRACT_PATH = "glove/"
    STORE_PATH = ''

    # Download file name

    FILE_PATH = "glove.zip"
    PRE_TRAIN_FILE_LINK = "http://nlp.stanford.edu/data/glove.6B.zip"
    PRE_TRAIN_FILE_NAME = "glove.6B.100d.txt"

    PRE_TRAIN_DETECTION_FILE_LINK = "https://s3.amazonaws.com/amdegroot-models/ssd300_mAP_77.43_v2.pth"

    VOICE_GENERATOR_MODELS = [
        GoogleDriveFile(google_drive_id='1E-B92LZz4dgg8DU81D6pyhOzM9yvvBTj', local_name='vg.pth')]
    VOICE_RECONGINIZER_MODELS = [
        GoogleDriveFile(google_drive_id='1RQQB-Yd-aqb6scWtnu1K4nlSTxTyaKjI', local_name='vr.pth')]
    FACE_DETECTOR_MODELS = [
        GoogleDriveFile(google_drive_id='1QJWKpAHRrAjrYPl6hQNDaoyBjoa_LRgz', local_name='pnet.pt'),
        GoogleDriveFile(google_drive_id='10aCiR393E6TLkp9KPPl4JhZamYqUVBO1', local_name='rnet.pt'),
        GoogleDriveFile(google_drive_id='1RRBtPlzw46peS-A8pyYGsPRHHFIUrSVV', local_name='onet.pt')]
    OBJECT_DETECTOR_MODELS = [
        GoogleDriveFile(google_drive_id='1QGG1trfj-z5_2OGNoSarUB4wx81cG-sa', local_name='oo.pth')]
    SENTIMENT_ANALYSIS_MODELS = [
        GoogleDriveFile(google_drive_id='1flRlQjfIa2toQ6HNmInhqrh4NuxGh8pT', local_name='sa.pth')]
    TOPIC_CLASSIFIER_MODELS = [
        GoogleDriveFile(google_drive_id='1U7C3xPid1ZvBKpkfW9KikrmNui0yJqnk', local_name='tc.pth')]
    PRETRAINED_VOCAB_BERT_BASE_UNCASED = \
        GoogleDriveFile(google_drive_id='1hlPkUSPeT5ZQBYZ1Z734BbnHIvpx2ZLj', local_name='vbbu.txt')
    PRETRAINED_VOCAB_BERT_BASE_CASED = \
        GoogleDriveFile(google_drive_id='1FLytUhOIF0mTfA4A9MtE3aQ1kJr96oTR', local_name='vbbc.txt')
    PRETRAINED_MODEL_BERT_BASE_UNCASED = \
        GoogleDriveFile(google_drive_id='1rp1rVBoQwqgvg-JE8JwLL-adgLE07oTG', local_name='mbbu.pth')
    PRETRAINED_MODEL_BERT_BASE_CASED = \
        GoogleDriveFile(google_drive_id='1YKoGj-e4zoyTabt5dYpgEPe-PAmjOTDV', local_name='mbbc.pth')

    VOICE_RECONGINIZER_LABELS = "_'ABCDEFGHIJKLMNOPQRSTUVWXYZ "
    VOICE_RECONGINIZER_AUDIO_CONF = {'sample_rate': 16000, 'window_size': 0.02, 'window_stride': 0.01,
                                     'window': 'hamming', 'noise_dir': None, 'noise_prob': 0.4,
                                     'noise_levels': (0.0, 0.5)}

    # Image Resize

    MAX_IMAGE_SIZE = 128 * 128

    # SYS Constant

    SYS_LINUX = 'linux'
    SYS_WINDOWS = 'windows'
    SYS_GOOGLE_COLAB = 'goog_colab'

    # Google drive downloader
    CHUNK_SIZE = 32768
    DOWNLOAD_URL = "https://docs.google.com/uc?export=download"
